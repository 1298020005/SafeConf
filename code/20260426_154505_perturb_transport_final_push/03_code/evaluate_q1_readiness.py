#!/usr/bin/env python3
"""Score a SafeTrans evidence folder against Q1 / Q2-top publication bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY_MODEL_CANDIDATES = [
    "PolicySafeTransPT",
    "DeepCalibratedSafeTransport",
    "TopRankGraftV2",
    "EffectBlendV2",
    "SafeTransPT",
    "NetworkSafeTransPT",
]
COMPARATOR_MODELS = ["V0", "V2", "ContextSimBaseline"]


def _pick_primary_model(summary: pd.DataFrame, requested: str | None = None) -> str:
    present = set(summary["model"].dropna().astype(str))
    if requested:
        if requested not in present:
            raise ValueError(f"Requested primary model {requested!r} not in {present}")
        return requested
    for name in PRIMARY_MODEL_CANDIDATES:
        if name in present:
            return name
    raise ValueError(f"No primary model in {present}")


def _load_summary(results_dir: Path) -> pd.DataFrame:
    for name in ("SAFETY_SUMMARY.csv", "GPU_DEEP_SUMMARY.csv", "FULL_SUMMARY_TABLE.csv"):
        path = results_dir / name
        if path.exists():
            df = pd.read_csv(path)
            if "phase" in df.columns:
                df = df.copy()
                df.loc[df["phase"].astype(str) == "gpu_deep", "phase"] = "main"
            return df
    raise FileNotFoundError(f"No summary table in {results_dir}")


def _compare(summary: pd.DataFrame, model: str, baseline: str) -> pd.DataFrame:
    rows = []
    idx = ["phase", "dataset", "split_type"]
    for keys, sub in summary.groupby(idx, dropna=False):
        cur = sub[sub["model"] == model]
        base = sub[sub["model"] == baseline]
        if cur.empty or base.empty:
            continue
        c, b = cur.iloc[0], base.iloc[0]
        row = dict(zip(idx, keys))
        row.update(
            {
                "model": model,
                "baseline": baseline,
                "top20_delta": c["top20_overlap_mean"] - b["top20_overlap_mean"],
                "deg_precision_delta": c["deg_precision_top50_mean"] - b["deg_precision_top50_mean"],
                "program_consistency_delta": c["program_shift_consistency_mean"]
                - b["program_shift_consistency_mean"],
                "rmse_delta": c["rmse_mean"] - b["rmse_mean"],
                "pearson_delta": c["pearson_mean"] - b["pearson_mean"],
            }
        )
        row["effect_win"] = int(
            (row["top20_delta"] >= 0.01 or row["deg_precision_delta"] >= 0.01)
            and (row["program_consistency_delta"] > 0 or row["rmse_delta"] <= 0.002)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _fraction_wins(deltas: pd.DataFrame, split_type: str | None = None) -> float:
    if deltas.empty:
        return 0.0
    sub = deltas
    if split_type:
        sub = sub[sub["split_type"] == split_type]
    if sub.empty:
        return 0.0
    return float(sub["effect_win"].mean())


def _risk_coverage_gain(risk_df: pd.DataFrame, model: str, coverage: float = 0.8) -> dict:
    if risk_df.empty:
        return {"status": "missing", "rmse_gain": 0.0}
    sub = risk_df[risk_df["model"] == model]
    if sub.empty or "coverage" not in sub.columns:
        return {"status": "missing", "rmse_gain": 0.0}
    full = sub[sub["coverage"] >= 0.99]
    partial = sub[sub["coverage"].between(coverage - 0.05, coverage + 0.05)]
    if full.empty or partial.empty:
        return {"status": "partial", "rmse_gain": 0.0}
    full_rmse = float(full["rmse"].mean())
    part_rmse = float(partial["rmse"].mean())
    gain = (full_rmse - part_rmse) / max(full_rmse, 1e-8)
    return {"status": "ok", "rmse_gain": gain, "full_rmse": full_rmse, "partial_rmse": part_rmse}


def _unsafe_contrast_ok(contrast_df: pd.DataFrame, model: str) -> float:
    if contrast_df.empty:
        return 0.0
    sub = contrast_df[(contrast_df["model"] == model) & (contrast_df["status"] == "ok")]
    if sub.empty or "unsafe_minus_safe_rmse" not in sub.columns:
        return 0.0
    return float((sub["unsafe_minus_safe_rmse"] > 0).mean())


def evaluate(results_dir: Path, primary_model_override: str | None = None) -> dict:
    summary = _load_summary(results_dir)
    primary_model = _pick_primary_model(summary, primary_model_override)
    risk_path = results_dir / "RISK_COVERAGE.csv"
    contrast_path = results_dir / "SAFE_UNSAFE_CONTRAST.csv"
    risk_df = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame()
    contrast_df = pd.read_csv(contrast_path) if contrast_path.exists() else pd.DataFrame()

    vs_v0 = _compare(summary, primary_model, "V0")
    vs_v2 = _compare(summary, primary_model, "V2")
    vs_ctx = _compare(summary, primary_model, "ContextSimBaseline")

    heldout_main = vs_v0[(vs_v0["phase"] == "main") & (vs_v0["split_type"] == "heldout_perturbation")]
    heldout_ext = vs_v0[(vs_v0["phase"] == "external") & (vs_v0["split_type"] == "heldout_perturbation")]
    leave_main = vs_v0[(vs_v0["phase"] == "main") & (vs_v0["split_type"] == "leave_context")]

    risk = _risk_coverage_gain(risk_df, primary_model)
    abstain_vs_full = (
        _compare(summary, "SafeTransPT", "SafeTransPT_no_abstain")
        if "SafeTransPT_no_abstain" in set(summary["model"].astype(str))
        else pd.DataFrame()
    )
    unsafe_frac = _unsafe_contrast_ok(contrast_df, primary_model)

    checks = {
        "heldout_main_win_frac": _fraction_wins(heldout_main),
        "heldout_main_n": int(len(heldout_main)),
        "heldout_ext_win_frac": _fraction_wins(heldout_ext),
        "heldout_ext_n": int(len(heldout_ext)),
        "leave_main_program_win_frac": float(
            (leave_main["program_consistency_delta"] > 0).mean() if len(leave_main) else 0.0
        ),
        "beats_v2_heldout_frac": _fraction_wins(
            vs_v2[vs_v2["split_type"] == "heldout_perturbation"] if not vs_v2.empty else vs_v2
        ),
        "beats_contextsim_heldout_frac": _fraction_wins(
            vs_ctx[vs_ctx["split_type"] == "heldout_perturbation"] if not vs_ctx.empty else vs_ctx
        ),
        "has_contextsim_baseline": bool(not vs_ctx.empty),
        "risk_rmse_gain_at_80cov": risk.get("rmse_gain", 0.0),
        "unsafe_contrast_ok_frac": unsafe_frac,
        "abstain_leave_rmse_better_frac": float(
            (
                abstain_vs_full[
                    (abstain_vs_full["split_type"] == "leave_context") & (abstain_vs_full["rmse_delta"] < 0)
                ].shape[0]
                / max(abstain_vs_full[abstain_vs_full["split_type"] == "leave_context"].shape[0], 1)
            )
            if not abstain_vs_full.empty
            else 0.0
        ),
    }

    q2_top_pass = (
        checks["heldout_main_win_frac"] >= 0.60
        and checks["heldout_main_n"] >= 3
        and checks["heldout_ext_win_frac"] >= 0.50
        and checks["heldout_ext_n"] >= 2
        and checks["risk_rmse_gain_at_80cov"] >= 0.03
        and checks["unsafe_contrast_ok_frac"] >= 0.50
    )

    contextsim_ok = checks["has_contextsim_baseline"] and checks["beats_contextsim_heldout_frac"] >= 0.70

    q1_pass = (
        q2_top_pass
        and checks["heldout_main_win_frac"] >= 0.75
        and checks["beats_v2_heldout_frac"] >= 0.55
        and contextsim_ok
        and checks["leave_main_program_win_frac"] >= 0.50
        and checks["heldout_ext_n"] >= 3
        and checks["heldout_ext_win_frac"] >= 0.65
    )

    if q1_pass:
        label = "Q1_READY_CANDIDATE"
    elif q2_top_pass:
        label = "Q2_TOP_READY"
    elif checks["heldout_main_win_frac"] >= 0.40:
        label = "Q2_CANDIDATE_NEEDS_CONFIRMATION"
    else:
        label = "NOT_READY"

    gaps = []
    if checks["heldout_main_win_frac"] < 0.75:
        gaps.append(f"{primary_model} must win V0 on >=75% main held-out settings (effect metrics).")
    if checks["beats_v2_heldout_frac"] < 0.55:
        gaps.append("Must beat V2 on >=55% held-out perturbation settings.")
    if checks["has_contextsim_baseline"] and checks["beats_contextsim_heldout_frac"] < 0.70:
        gaps.append("Must beat ContextSimBaseline on >=70% held-out settings.")
    if not checks["has_contextsim_baseline"]:
        gaps.append("ContextSimBaseline not in summary — run 46_q1_cpu_push to add reviewer comparator.")
    if checks["heldout_ext_n"] < 3:
        gaps.append("Need >=3 external datasets with held-out perturbation results.")
    if checks["risk_rmse_gain_at_80cov"] < 0.03:
        gaps.append("Risk-coverage: RMSE at ~80% coverage should improve >=3% vs full coverage.")
    if checks["unsafe_contrast_ok_frac"] < 0.50:
        gaps.append("Unsafe tasks should have higher RMSE than safe tasks in >=50% settings.")

    report = {
        "results_dir": str(results_dir),
        "label": label,
        "primary_model": primary_model,
        "checks": checks,
        "gaps": gaps,
        "n_summary_rows": int(len(summary)),
        "models_present": sorted(summary["model"].dropna().unique().tolist()),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Folder with SAFETY_SUMMARY or GPU_DEEP_SUMMARY")
    parser.add_argument("--primary-model", default=None, help="Optional model override for ablation/probe scoring")
    parser.add_argument("--write-md", action="store_true")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    report = evaluate(results_dir, args.primary_model)
    suffix = f"_{args.primary_model}" if args.primary_model else ""
    out_json = results_dir / f"Q1_READINESS_REPORT{suffix}.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.write_md:
        lines = [
            "# Q1 / Q2-top readiness report",
            "",
            f"- **Label:** `{report['label']}`",
            f"- **Primary model:** `{report['primary_model']}`",
            "",
            "## Checks",
            "",
        ]
        for k, v in report["checks"].items():
            if isinstance(v, float):
                lines.append(f"- `{k}`: {v:.3f}")
            else:
                lines.append(f"- `{k}`: {v}")
        if report["gaps"]:
            lines.extend(["", "## Gaps to close", ""])
            for g in report["gaps"]:
                lines.append(f"- {g}")
        (results_dir / f"Q1_READINESS_REPORT{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

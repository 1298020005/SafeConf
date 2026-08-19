from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push")


def collect(results: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in results.rglob(pattern):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["source_file"] = str(path)
        parts = path.relative_to(results).parts
        df["run_group"] = parts[0] if parts else ""
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def best_rows(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    if df.empty or model_col not in df:
        return pd.DataFrame()
    metric_cols = [c for c in ["top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"] if c in df]
    if not metric_cols:
        return pd.DataFrame()
    out = df.copy()
    out["effect_signal_score"] = 0.0
    for col in ["top20_delta", "deg_precision_delta", "program_consistency_delta"]:
        if col in out:
            out["effect_signal_score"] += out[col].fillna(0)
    return out.sort_values("effect_signal_score", ascending=False).head(30)


def plain_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return df.to_string(index=False, max_colwidth=28)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", default=str(ROOT / "16_six_day_offline_push"))
    args = parser.parse_args()
    push = Path(args.push)
    results = push / "results"
    reports = push / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    safetrans = collect(results, "SAFETRANS_VS_FIXED_V2_DELTAS.csv")
    gpu = collect(results, "GPU_DEEPSAFE_VS_V0_*.csv")
    gpu_vs_v2 = collect(results, "GPU_DEEPSAFE_VS_V2_*.csv")
    statuses = []
    for path in results.rglob("*STATUS*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data["source_file"] = str(path)
        statuses.append(data)
    status_df = pd.DataFrame(statuses)

    if not safetrans.empty:
        safetrans.to_csv(reports / "ALL_SAFETRANS_VS_FIXED_V2.csv", index=False)
        best_rows(safetrans).to_csv(reports / "BEST_SAFETRANS_EFFECT_SIGNALS.csv", index=False)
    if not gpu.empty:
        gpu.to_csv(reports / "ALL_GPU_DEEPSAFE_VS_V0.csv", index=False)
        best_rows(gpu).to_csv(reports / "BEST_GPU_EFFECT_SIGNALS.csv", index=False)
    if not gpu_vs_v2.empty:
        gpu_vs_v2.to_csv(reports / "ALL_GPU_DEEPSAFE_VS_V2.csv", index=False)
        best_rows(gpu_vs_v2).to_csv(reports / "BEST_GPU_EFFECT_SIGNALS_VS_V2.csv", index=False)
    if not status_df.empty:
        status_df.to_csv(reports / "ALL_STATUS_SUMMARY.csv", index=False)

    lines = [
        "# Six-day offline push status",
        "",
        f"SafeTrans delta tables: {len(safetrans)} rows",
        f"GPU DeepSafe delta tables: {len(gpu)} rows",
        f"GPU DeepSafe vs V2 delta tables: {len(gpu_vs_v2)} rows",
        f"Status files: {len(status_df)}",
        "",
        "## Current best SafeTrans rows",
        "",
    ]
    if not safetrans.empty:
        cols = [c for c in ["run_group", "phase", "dataset", "split_type", "top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"] if c in safetrans]
        lines.append("```")
        lines.append(plain_table(best_rows(safetrans)[cols].head(10)))
        lines.append("```")
    else:
        lines.append("No SafeTrans result table found yet.")
    lines += ["", "## Current best GPU rows", ""]
    if not gpu.empty:
        cols = [c for c in ["run_group", "phase", "dataset", "split_type", "top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"] if c in gpu]
        lines.append("```")
        lines.append(plain_table(best_rows(gpu)[cols].head(10)))
        lines.append("```")
    else:
        lines.append("No GPU result table found yet.")
    lines += ["", "## Current best GPU rows vs fixed V2", ""]
    if not gpu_vs_v2.empty:
        cols = [c for c in ["run_group", "phase", "dataset", "split_type", "top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"] if c in gpu_vs_v2]
        lines.append("```")
        lines.append(plain_table(best_rows(gpu_vs_v2)[cols].head(10)))
        lines.append("```")
    else:
        lines.append("No GPU vs V2 result table found yet. Newly launched GPU cycles will produce this table.")
    (reports / "SIX_DAY_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(reports / "SIX_DAY_STATUS.md")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [
        "top20_delta",
        "deg_precision_delta",
        "program_consistency_delta",
        "network_module_consistency_delta",
        "pearson_delta",
        "spearman_delta",
    ]
    cols = [c for c in cols if c in df]
    rows = []
    for keys, sub in df.groupby(["phase", "split_type"], dropna=False):
        row = dict(zip(["phase", "split_type"], keys))
        row["n"] = len(sub)
        row["run_groups"] = sub["run_group"].nunique() if "run_group" in sub else 0
        for col in cols:
            row[f"{col}_mean"] = float(sub[col].mean())
        if {"top20_delta", "deg_precision_delta", "program_consistency_delta"}.issubset(sub.columns):
            row["two_plus_effect_fraction"] = float(
                ((sub[["top20_delta", "deg_precision_delta", "program_consistency_delta"]] > 0).sum(axis=1) >= 2).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phase", "split_type"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", required=True)
    args = parser.parse_args()
    push = Path(args.push)
    reports = push / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    results = push / "results"

    vs_v2 = collect(results, "NetworkSafeTransPT_VS_V2.csv")
    no_abstain = collect(results, "NetworkSafeTransPT_no_abstain_VS_V2.csv")
    vs_v0 = collect(results, "NetworkSafeTransPT_VS_V0.csv")
    statuses = []
    for path in results.rglob("NETWORK_SAFETRANS_STATUS.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data["source_file"] = str(path)
        parts = path.relative_to(results).parts
        data["run_group"] = parts[0] if parts else ""
        statuses.append(data)
    status_df = pd.DataFrame(statuses)

    if not vs_v2.empty:
        vs_v2.to_csv(reports / "ALL_NETWORK_SAFE_VS_V2.csv", index=False)
        metric_summary(vs_v2).to_csv(reports / "NETWORK_SAFE_VS_V2_BY_SETTING.csv", index=False)
    if not no_abstain.empty:
        no_abstain.to_csv(reports / "ALL_NETWORK_NO_ABSTAIN_VS_V2.csv", index=False)
    if not vs_v0.empty:
        vs_v0.to_csv(reports / "ALL_NETWORK_SAFE_VS_V0.csv", index=False)
    if not status_df.empty:
        status_df.to_csv(reports / "NETWORK_STATUS_FILES.csv", index=False)

    lines = [
        "# Network-aware SafeTrans-PT status",
        "",
        f"NetworkSafeTransPT vs V2 rows: {len(vs_v2)}",
        f"NetworkSafeTransPT no-abstain vs V2 rows: {len(no_abstain)}",
        f"NetworkSafeTransPT vs V0 rows: {len(vs_v0)}",
        f"Status files: {len(status_df)}",
        "",
        "## Mean deltas vs fixed V2 by setting",
        "",
    ]
    summary = metric_summary(vs_v2)
    if summary.empty:
        lines.append("No completed network result table yet.")
    else:
        lines.append("```")
        lines.append(summary.to_string(index=False, max_colwidth=26))
        lines.append("```")
    if not vs_v2.empty:
        cols = [
            c
            for c in [
                "run_group",
                "phase",
                "dataset",
                "split_type",
                "top20_delta",
                "deg_precision_delta",
                "program_consistency_delta",
                "network_module_consistency_delta",
                "pearson_delta",
                "spearman_delta",
            ]
            if c in vs_v2
        ]
        top = vs_v2.copy()
        score_cols = [c for c in ["top20_delta", "deg_precision_delta", "program_consistency_delta", "network_module_consistency_delta"] if c in top]
        top["network_effect_sum"] = top[score_cols].sum(axis=1)
        lines += ["", "## Best rows", "", "```", top.sort_values("network_effect_sum", ascending=False)[cols].head(12).to_string(index=False, max_colwidth=24), "```"]
    (reports / "NETWORK_PUSH_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(reports / "NETWORK_PUSH_STATUS.md")


if __name__ == "__main__":
    main()

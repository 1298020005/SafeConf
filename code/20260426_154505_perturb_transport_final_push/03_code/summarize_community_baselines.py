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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", required=True)
    args = parser.parse_args()
    push = Path(args.push)
    reports = push / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    results = push / "results"

    summary = collect(results, "COMMUNITY_BASELINE_SUMMARY.csv")
    deltas = []
    for pattern in ["*_VS_V0.csv", "*_VS_V2.csv"]:
        for path in results.rglob(pattern):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            df["source_file"] = str(path)
            parts = path.relative_to(results).parts
            df["run_group"] = parts[0] if parts else ""
            deltas.append(df)
    delta_df = pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame()
    statuses = []
    for path in results.rglob("COMMUNITY_BASELINE_STATUS.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data["source_file"] = str(path)
        parts = path.relative_to(results).parts
        data["run_group"] = parts[0] if parts else ""
        statuses.append(data)
    status_df = pd.DataFrame(statuses)

    if not summary.empty:
        summary.to_csv(reports / "ALL_COMMUNITY_BASELINE_SUMMARY.csv", index=False)
    if not delta_df.empty:
        delta_df.to_csv(reports / "ALL_COMMUNITY_BASELINE_DELTAS.csv", index=False)
        key_cols = ["top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"]
        rows = []
        for keys, sub in delta_df.groupby(["baseline", "model", "phase", "split_type"], dropna=False):
            row = dict(zip(["baseline", "model", "phase", "split_type"], keys))
            row["n"] = len(sub)
            for col in key_cols:
                if col in sub:
                    row[col] = float(sub[col].mean())
            if {"top20_delta", "deg_precision_delta", "program_consistency_delta"}.issubset(sub.columns):
                row["two_plus_effect_fraction"] = float(
                    ((sub[["top20_delta", "deg_precision_delta", "program_consistency_delta"]] > 0).sum(axis=1) >= 2).mean()
                )
            rows.append(row)
        pd.DataFrame(rows).to_csv(reports / "COMMUNITY_BASELINE_BY_SETTING.csv", index=False)
    if not status_df.empty:
        status_df.to_csv(reports / "COMMUNITY_BASELINE_STATUS_FILES.csv", index=False)

    lines = [
        "# Community-inspired baseline status",
        "",
        f"Summary rows: {len(summary)}",
        f"Delta rows: {len(delta_df)}",
        f"Status files: {len(status_df)}",
        "",
    ]
    by = reports / "COMMUNITY_BASELINE_BY_SETTING.csv"
    if by.exists():
        df = pd.read_csv(by)
        lines += ["## Mean deltas by setting", "", "```", df.to_string(index=False, max_colwidth=24), "```"]
    else:
        lines.append("No completed delta table yet.")
    (reports / "COMMUNITY_BASELINE_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(reports / "COMMUNITY_BASELINE_STATUS.md")


if __name__ == "__main__":
    main()

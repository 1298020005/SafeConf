#!/usr/bin/env python3
"""Create a historical run copy with blank perturbation rows removed.

The source run is never modified. Array stores are copied unchanged because
extra, unreferenced array keys are valid; only row-oriented CSV tables are
filtered. The output is contract-checked before it is accepted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from safetrans_confidence.data.records import validate_prediction_record_artifacts

ROW_TABLES = (
    "PREDICTION_RECORDS.csv",
    "CONFIDENCE_FEATURES.csv",
    "CONFIDENCE_SCORES.csv",
)
BLANK_STRINGS = {"", "nan", "none", "null"}


def _valid_perturbation(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.notna() & ~text.str.lower().isin(BLANK_STRINGS)


def build_drop_blank_run(source_dir: Path, out_dir: Path) -> dict:
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out_dir}")

    shutil.copytree(source_dir, out_dir)
    log_rows: list[dict] = []
    for name in ROW_TABLES:
        path = out_dir / "tables" / name
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "perturbation" not in table.columns:
            raise ValueError(f"{name} has no perturbation column")
        keep = _valid_perturbation(table["perturbation"])
        filtered = table.loc[keep].copy()
        filtered.to_csv(path, index=False)
        log_rows.append(
            {
                "table": name,
                "rows_before": int(len(table)),
                "rows_after": int(len(filtered)),
                "dropped_rows": int((~keep).sum()),
            }
        )

    issues = validate_prediction_record_artifacts(
        out_dir,
        strict=False,
        require_effect_arrays=True,
    )
    if issues:
        raise ValueError("sanitized run still violates contract: " + ";".join(issues))

    pd.DataFrame(log_rows).to_csv(out_dir / "DROP_BLANK_LOG.csv", index=False)
    status = {
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "status": "ok",
        "tables": log_rows,
    }
    (out_dir / "DROP_BLANK_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a historical run and remove blank perturbation rows."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_drop_blank_run(args.source_dir, args.out_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

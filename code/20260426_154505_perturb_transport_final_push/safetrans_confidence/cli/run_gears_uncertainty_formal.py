from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetrans_confidence.predictors.gears_uncertainty_export import export_gears_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal GEARS uncertainty/proxy export.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/gears_prediction_records_formal"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/gears_uncertainty_formal"),
    )
    args = parser.parse_args()
    print(json.dumps(export_gears_uncertainty(args.input_root, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


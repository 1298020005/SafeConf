#!/usr/bin/env python3
"""Confidence MVP for KaggleCrossPatient only (blind chem_robust probe)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "confidence_task"
os.chdir(CONF_DIR)
sys.path.insert(0, str(CONF_DIR))
sys.path.insert(0, str(ROOT))

import run_confidence_mvp_v2_1 as mvp  # noqa: E402

mvp.DATASET_NAMES = ["KaggleCrossPatient"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--atlas-root", default=str(mvp.DEFAULT_ATLAS_ROOT))
    parser.add_argument("--n-genes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=5201)
    args = parser.parse_args()
    sys.argv = [
        "run_kcp_blind_probe",
        "--project-root",
        args.project_root,
        "--atlas-root",
        args.atlas_root,
        "--out-dir",
        args.out_dir,
        "--n-genes",
        str(args.n_genes),
        "--seed",
        str(args.seed),
    ]
    return int(mvp.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

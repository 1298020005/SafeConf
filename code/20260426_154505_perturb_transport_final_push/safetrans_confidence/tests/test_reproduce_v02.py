from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V02_REF = PROJECT_ROOT / "outputs" / "confidence_task_protocol_v0_2" / "tables" / "PROTOCOL_V0_2_EVAL_SUMMARY.csv"
PKG_OUT = PROJECT_ROOT / "outputs" / "benchmark_protocol_v0_2_pkg" / "tables" / "CONFIDENCE_EVAL_SUMMARY.csv"

TARGETS = {
    "Haber": 0.4645,
    "Parekh": 0.5556,
    "KaggleCrossCell": 0.4421,
}
TOL = 0.05


@pytest.mark.skipif(not PKG_OUT.exists(), reason="run cli first")
def test_reproduce_protocol_rho_within_tolerance():
    eval_df = pd.read_csv(PKG_OUT)
    sub = eval_df[
        (eval_df["level"] == "dataset")
        & (eval_df["score_name"] == "protocol_v0_2_family_confidence")
    ]
    for dataset, target in TARGETS.items():
        row = sub[sub["dataset_name"] == dataset]
        assert len(row) == 1, dataset
        rho = float(row["direction_aligned_spearman"].iloc[0])
        assert abs(rho - target) <= TOL, f"{dataset}: {rho} vs {target}"


@pytest.mark.skipif(not (V02_REF.exists() and PKG_OUT.exists()), reason="refs missing")
def test_matches_legacy_v02_script():
    ref = pd.read_csv(V02_REF)
    new = pd.read_csv(PKG_OUT)
    # Family-level rows share dataset_name="ALL"; include the family to avoid
    # a Cartesian merge between gene_main and chem_robust reference rows.
    id_columns = ["level", "dataset_family", "dataset_name", "score_name"]
    key = id_columns + ["direction_aligned_spearman"]
    ref_sub = ref[ref["score_name"] == "protocol_v0_2_family_confidence"][key]
    new_sub = new[new["score_name"] == "protocol_v0_2_family_confidence"][key]
    merged = ref_sub.merge(new_sub, on=id_columns, suffixes=("_ref", "_new"))
    for _, r in merged.iterrows():
        assert abs(r["direction_aligned_spearman_ref"] - r["direction_aligned_spearman_new"]) < 1e-6

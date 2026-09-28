#!/usr/bin/env python3
"""Apply the preregistered development gate and freeze one Final Candidate."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
DEV = STAGE / "safeconf_v4_development"
CONFIG = STAGE / "FINAL_METHOD_CONFIG.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text())
    if config.get("final_candidate") is not None:
        raise FileExistsError("Final Candidate already frozen")
    pieces = []
    gates = {}
    for name in ("txpert_gat", "txpert_exphormer"):
        result = pd.read_csv(DEV / name / "STRATUM_RESULTS.csv")
        pieces.append(result.assign(upstream=name))
        gates[name] = json.loads((DEV / name / "DEVELOPMENT_GATE.json").read_text())
    strata = pd.concat(pieces, ignore_index=True)
    pivot = strata.pivot(index=["upstream", "target"], columns="method", values="utility20")
    deltas = pivot["V2_nested"] - pivot["Ridge_USR"]
    summary_parts = []
    for name in ("txpert_gat", "txpert_exphormer"):
        summary_parts.append(pd.read_csv(DEV / name / "SUMMARY.csv").assign(upstream=name))
    summary = pd.concat(summary_parts, ignore_index=True)
    means = summary.groupby("method").agg({
        "utility20": "mean", "risk_at_10": "mean", "risk_at_20": "mean",
        "risk_at_50": "mean", "high_risk_miss_rate": "mean", "aurc": "mean",
    })
    def rel_deg(column: str) -> float:
        v1 = float(means.loc["Ridge_USR", column]); v2 = float(means.loc["V2_nested", column])
        return max(0.0, (v2 - v1) / max(abs(v1), 1e-12))
    aggregate = {
        "delta_u20_macro": float(means.loc["V2_nested", "utility20"] - means.loc["Ridge_USR", "utility20"]),
        "nonnegative_strata_fraction": float((deltas >= 0).mean()),
        "risk10_relative_degradation": rel_deg("risk_at_10"),
        "risk20_relative_degradation": rel_deg("risk_at_20"),
        "risk50_relative_degradation": rel_deg("risk_at_50"),
        "high_risk_miss_rate_degradation": max(0.0, float(means.loc["V2_nested", "high_risk_miss_rate"] - means.loc["Ridge_USR", "high_risk_miss_rate"])),
        "aurc_relative_degradation": rel_deg("aurc"),
        "valid_strata_fraction": float(deltas.notna().mean()),
        "n_evaluation_strata": int(len(deltas)),
    }
    passed = bool(
        aggregate["delta_u20_macro"] >= 0.005
        and aggregate["nonnegative_strata_fraction"] >= 0.60
        and max(aggregate["risk10_relative_degradation"], aggregate["risk20_relative_degradation"], aggregate["risk50_relative_degradation"]) <= 0.05
        and aggregate["high_risk_miss_rate_degradation"] <= 0.02
        and aggregate["aurc_relative_degradation"] <= 0.05
        and aggregate["valid_strata_fraction"] >= 0.80
        and all(g["v2_passes_development_gate"] for g in gates.values())
    )
    candidate = "V2_nested_evidence_shrinkage" if passed else "V1_Ridge_UniversalP_Support_Relevance"
    config["status"] = "FINAL_CANDIDATE_FROZEN_PRE_CONFIRMATION"
    config["final_candidate"] = candidate
    config["final_candidate_frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["development_gate_result"] = aggregate
    config["per_upstream_gate_result"] = gates
    config["final_candidate_features"] = {
        "prediction_only": [
            "predicted_magnitude", "prediction_abs_mean", "prediction_signed_mean",
            "prediction_std", "prediction_abs_q95", "prediction_sparsity",
        ],
        "support": ["n_source_cells", "n_source_contexts", "n_source_batches", "min_source_cells"],
        "relevance": ["prediction_source_cosine", "negative_model_source_gap"],
        "quality": [],
        "conflict_proxy": ["source_delta_dispersion", "conflict_missing"],
        "content": ["source_transfer_magnitude"],
    }
    config["limitations_at_freeze"] = [
        "No legal replicate/split-half Quality field exists in the TxPert development assets.",
        "Conflict is represented by source-effect dispersion and is not a pure measurement-quality variable.",
        "Development bootstrap CI for the V2-V1 Utility@20 increment crosses zero in each architecture; independent confirmation remains required.",
        "TxPert GAT and Exphormer are cross-architecture evidence within one upstream family.",
    ]
    code = ROOT / "tools/scripts/run_safeconf_v4_development.py"
    config["implementation_sha256"] = sha(code)
    config["git_head_before_freeze_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    freeze = {
        "candidate": candidate,
        "passed_v2_gate": passed,
        "aggregate": aggregate,
        "per_upstream": gates,
        "config_sha256": sha(CONFIG),
        "implementation_sha256": sha(code),
        "confirmation_opened_before_freeze": False,
    }
    (STAGE / "FINAL_CANDIDATE_FREEZE.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# SafeConf Final Candidate Freeze",
        "",
        f"Final Candidate: **{candidate}**.",
        "",
        "This decision was made from DEV/SEEN only. No SEALED feature distribution, prediction, truth or result was opened.",
        "",
        "## Registered aggregate development gate",
        "",
        *[f"- `{key}`: `{value}`" for key, value in aggregate.items()],
        "",
        "## Interpretation",
        "",
        "V2 passed the preregistered practical gate across both TxPert architectures. The paired bootstrap interval for its incremental Utility@20 still crosses zero, so this freeze selects the confirmation candidate; it does not assert independent confirmation.",
        "",
        "Quality is absent rather than imputed. V2 uses Support, Relevance, a Conflict proxy, Content and Missingness, with monotonic evidence weights.",
    ]
    (STAGE / "FINAL_CANDIDATE_FREEZE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

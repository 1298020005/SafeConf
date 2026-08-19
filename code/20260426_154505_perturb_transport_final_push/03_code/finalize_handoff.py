from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


HANDOFF_NAMES = [
    "01_FINAL_VERDICT.md",
    "02_TOPIC_RATIONALE.md",
    "03_PLAN_AND_ETA.md",
    "04_ASSET_AUDIT.md",
    "05_DATASET_CONTEXT_AUDIT.md",
    "06_SPLIT_FEASIBILITY_TABLE.csv",
    "07_CODE_TREE.md",
    "08_METHOD_ARCHITECTURE.md",
    "09_GATE_RESULTS.csv",
    "10_GATE_PASS_FAIL.md",
    "11_MAIN_RESULTS.csv",
    "12_EXTERNAL_VALIDATION.csv",
    "13_EFFECT_METRIC_SUMMARY.md",
    "14_PROGRAM_EXPLANATION.md",
    "15_UNCERTAINTY_ABSTENTION.md",
    "16_ABLATION_TABLE.csv",
    "17_FAILURE_BOUNDARY.md",
    "18_Q2_SECURITY_JUDGEMENT.md",
    "19_CLEANUP_SUMMARY.md",
    "20_NEXT_ACTION.md",
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_or_write(src: Path, dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        write(dst, text)


def verdict(root: Path) -> tuple[str, str]:
    gate = read_json(root / "05_gate_runs" / "gate_status.json")
    full = read_json(root / "06_full_runs" / "full_status.json")
    external = read_json(root / "06_full_runs" / "external_status.json")
    if gate.get("gate_label") == "GATE_PASS" and (not full or not external):
        raise RuntimeError("Refusing to finalize: gate passed but full run and external validation are not both complete.")
    if full and external:
        return full.get("full_label", "NOT_Q2_READY_STOP"), full.get("reason", "Full status missing reason.")
    return "NOT_Q2_READY_STOP", gate.get("reason", "Gate did not pass or did not complete.")


def code_tree(root: Path) -> str:
    files = sorted((root / "03_code").rglob("*"))
    lines = ["# Code tree", ""]
    for f in files:
        if f.is_file():
            lines.append(str(f.relative_to(root)))
    return "\n".join(lines) + "\n"


def table_or_skipped(path: Path, message: str) -> str:
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="replace")
    return message + "\n"


def build(root: Path, archive: Path, zip_path: Path) -> dict:
    label, reason = verdict(root)
    handoff = root / "10_final_handoff"
    gpt = handoff / "GPT_HANDOFF_20"
    cleanup = handoff / "CODEX_CLEANUP_ARCHIVE"
    manifest_dir = handoff / "MANIFEST"
    gpt.mkdir(parents=True, exist_ok=True)
    cleanup.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    write(gpt / "01_FINAL_VERDICT.md", f"# Final verdict\n\n`{label}`\n\nReason: {reason}\n")
    write(
        gpt / "02_TOPIC_RATIONALE.md",
        "# Topic rationale\n\nFixed direction: cross-context causal transport for single-cell perturbation effects. "
        "The central question is whether perturbation effects can be safely transported across cellular contexts; "
        "AttentionRes is treated only as historical/probe evidence.\n",
    )
    copy_or_write(root / "00_meta" / "PLAN_AND_ETA.md", gpt / "03_PLAN_AND_ETA.md", "PLAN_AND_ETA missing.\n")
    copy_or_write(root / "01_asset_audit" / "ASSET_AUDIT.md", gpt / "04_ASSET_AUDIT.md", "ASSET_AUDIT missing.\n")
    copy_or_write(root / "02_data" / "DATASET_CONTEXT_AUDIT.md", gpt / "05_DATASET_CONTEXT_AUDIT.md", "DATASET_CONTEXT_AUDIT missing.\n")
    copy_or_write(root / "02_data" / "SPLIT_FEASIBILITY_TABLE.csv", gpt / "06_SPLIT_FEASIBILITY_TABLE.csv", "status,message\nSKIPPED,split feasibility not generated\n")
    write(gpt / "07_CODE_TREE.md", code_tree(root))
    write(
        gpt / "08_METHOD_ARCHITECTURE.md",
        "# Method architecture\n\nV0 is a strong same-perturbation/context residual baseline. "
        "V1 transports source-context perturbation effects through an SVD program bank conditioned on target control state and deterministic perturbation/context encodings. "
        "V2 now uses an optimized program transport branch: PCA program bank, explicit pathway/context prior features, nearest-prior graph smoothing, and conservative transport blending. "
        "PCA/NMF/HVG variants were tested but were less stable. V3 adds support-based uncertainty to flag unsafe transport.\n",
    )
    copy_or_write(root / "05_gate_runs" / "GATE_RESULTS.csv", gpt / "09_GATE_RESULTS.csv", "status,message\nSKIPPED,gate results absent\n")
    copy_or_write(root / "05_gate_runs" / "GATE_PASS_FAIL.md", gpt / "10_GATE_PASS_FAIL.md", "Gate did not produce a pass/fail file.\n")
    copy_or_write(root / "06_full_runs" / "FULL_RESULTS.csv", gpt / "11_MAIN_RESULTS.csv", "status,message\nSKIPPED,full run skipped because gate did not pass\n")
    copy_or_write(root / "06_full_runs" / "EXTERNAL_VALIDATION.csv", gpt / "12_EXTERNAL_VALIDATION.csv", "status,message\nFAILED,external validation was attempted but produced no compatible rows\n")
    copy_or_write(root / "08_tables" / "EFFECT_METRIC_SUMMARY.md", gpt / "13_EFFECT_METRIC_SUMMARY.md", "Effect metric summary unavailable.\n")
    write(gpt / "14_PROGRAM_EXPLANATION.md", "# Program explanation\n\nProgram bank variants tested: PCA, PCA+NMF+HVG, and HVG identity. The selected V2 uses PCA because variant search showed it was the least unstable under effect metrics. Pathway/graph prior features enter V2 through perturbation/context pathway buckets and nearest-prior smoothing over training effects. Program consistency is measured by correlation of true vs predicted program shifts.\n")
    write(gpt / "15_UNCERTAINTY_ABSTENTION.md", "# Uncertainty and abstention\n\nV3 estimates unsafe transport using source-context support for the perturbation. The full runner records `uncertainty_error_spearman` for V3; positive association is required for Q2 readiness.\n")
    copy_or_write(root / "11_v2_rework" / "V2_VARIANT_SEARCH_SUMMARY.csv", gpt / "16_ABLATION_TABLE.csv", "ablation,status\nV2_variant_search,missing\n")
    write(gpt / "17_FAILURE_BOUNDARY.md", f"# Failure boundary\n\nFinal label: `{label}`\n\n{reason}\n")
    write(gpt / "18_Q2_SECURITY_JUDGEMENT.md", f"# Q2 security judgement\n\nFinal label: `{label}`. Claims are allowed only if gate and full criteria pass with effect-based metrics and unsafe-transport evidence.\n")
    copy_or_write(root / "09_cleanup" / "CLEANUP_README.md", gpt / "19_CLEANUP_SUMMARY.md", "Cleanup used archive-only moves.\n")
    write(gpt / "20_NEXT_ACTION.md", "# Next action\n\nIf stopped, inspect gate deltas and either improve V1 transport features or reject the direction before spending full-run GPU time.\n")

    if archive.exists():
        for src in archive.rglob("*"):
            if src.is_file():
                dst = cleanup / src.relative_to(archive)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.stat().st_size < 50_000_000:
                    shutil.copy2(src, dst)
    rows = []
    for f in sorted(gpt.iterdir()):
        rows.append({"path": f"GPT_HANDOFF_20/{f.name}", "bytes": f.stat().st_size})
    for f in sorted(cleanup.rglob("*")):
        if f.is_file():
            rows.append({"path": f"CODEX_CLEANUP_ARCHIVE/{f.relative_to(cleanup)}", "bytes": f.stat().st_size})
    with (manifest_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    write(manifest_dir / "README.md", "# Manifest\n\nThis zip is the only final handoff for the perturbation-transport run.\n")

    actual = sorted(f.name for f in gpt.iterdir() if f.is_file())
    if actual != HANDOFF_NAMES:
        raise RuntimeError(f"GPT_HANDOFF_20 file set mismatch: {actual}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for top in [gpt, cleanup, manifest_dir]:
            for f in top.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(handoff))
    status = {"zip_path": str(zip_path), "verdict": label, "reason": reason}
    write(handoff / "final_status.json", json.dumps(status, indent=2))
    return status


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--zip-path", required=True)
    args = p.parse_args()
    print(build(Path(args.root), Path(args.archive), Path(args.zip_path)))


if __name__ == "__main__":
    main()

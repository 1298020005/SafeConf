#!/usr/bin/env python3
"""Render saved Source control summaries after an optional-dependency failure.

No fitting, scoring, statistical recomputation, data selection, or RNG occurs.
The original runner and FAILED receipts remain byte-identical.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001"
D = R / "source_biology_label_control_v1"
O = Path("/home/yyf/runtime_artifacts/safeconf_research_20261001/source_biology_label_control_v1")
RUNNER = ROOT / "tools/scripts/run_safeconf_source_biology_label_control_v1.py"
PINS = {
    RUNNER: "5c10dbd21bcc4ee2965920ed5a0621a958be02b9b7dec08347f64c8d56a54b68",
    O / "STATUS.json": "c1afb2e289192129ac234ba5d89130b398e6fdf569259f2f5eea956f84007698",
    O / "SCORE_FREEZE.json": "9ed57c2430b35c40365a7bb10a25090e355b420e8525b7f2893b903853999f41",
    O / "SCORE_ONLY_PREDICTIONS.csv.gz": "30f1985886921f76187a9b6283efa0406a1914a8493887d02779c8eb1d23c506",
    O / "TASK_PREDICTIONS.csv.gz": "58b29d900f5b533dae18921f2fd80a5d011d7d381dc8a3db7a45ce263d932849",
    O / "ALL_FIXED_SOURCE_METRIC_DRAWS.npz": "4089da5ec760054a4c728fc4205c446e3d4d62fcbb1518f543ecdbbcc8b5a9c3",
}
TABLES = tuple(D / name for name in ("PAIRED_METRICS.csv", "METHOD_METRICS.csv", "LINE_MACRO_METRICS.csv",
    "FIT_COSTS.csv", "ORIGINAL_SCORE_REPLAY.csv", "BIOLOGICAL_CDF_AUDIT.csv", "FOLD_CONTEXT_METRICS.csv",
    "SCALAR_COUNTER_REPRODUCTION.csv", "CONFIG.json", "ACTUAL_STATUS.json", "MODEL_SCORE_HASH_RECEIPT.json"))
METRICS = ["utility20", "spearman", "aurc", "high_risk_miss_rate", "error_at_10", "error_at_20", "error_at_50"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""): h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))


def main():
    summary, receipt = D / "SUMMARY.md", D / "RENDER_ONLY_RECOVERY.json"
    if summary.exists() or receipt.exists(): raise RuntimeError("Fresh report-only recovery outputs required")
    before = {str(p): sha(p) for p in tuple(PINS) + TABLES}
    if any(before[str(p)] != expected for p, expected in PINS.items()): raise RuntimeError("Original result/code pin differs")
    status = json.loads((O / "STATUS.json").read_text())
    if (status["status"] != "FAILED" or "tabulate" not in status["failure"] or status["new_fits"] != 10
            or status["all_inputs_unchanged"] is not True): raise RuntimeError("Unexpected original failure")
    frozen = json.loads((O / "SCORE_FREEZE.json").read_text())
    if len(frozen["model_sha256"]) != 20: raise RuntimeError("Twenty sealed models required")
    for path, expected in frozen["model_sha256"].items():
        if sha(path) != expected: raise RuntimeError("Saved model bytes differ")
        before[path] = expected
    pairs = read_csv(D / "PAIRED_METRICS.csv"); macro = read_csv(D / "LINE_MACRO_METRICS.csv")
    if len(pairs) != 140 or len(macro) != 8: raise RuntimeError("Complete saved summaries required")
    main_rows = [r for r in pairs if r["context"] == "macro" and r["metric"] == "utility20" and r["method_a"] == "BioNorm_HGB"]
    if len(main_rows) != 2: raise RuntimeError("Both fixed primary contrasts required")
    text = "# Source biological label control: actual saved results\n\n"
    text += "The ten fits, twenty saved models and all fixed-count statistics completed. The original process exited with code 1 only while formatting this summary because `tabulate` was absent. Its FAILED receipt and exact runner are preserved. This renderer uses saved summaries only: zero new fits, scores, statistics or RNG.\n\n"
    text += "| Direction | BioNorm − OldRank U20 | Nominal paired 95% CI | Valid draws |\n| --- | ---: | --- | ---: |\n"
    for row in main_rows:
        text += f"| {row['line']} | {float(row['difference_a_minus_b']):+.6f} | [{float(row['ci95_lower']):+.6f}, {float(row['ci95_upper']):+.6f}] | {row['valid_draws']} / {row['saved_draws']} |\n"
    text += "\n| Direction | Method | " + " | ".join(METRICS) + " |\n| --- | --- | " + " | ".join(["---:"] * 7) + " |\n"
    for row in macro:
        text += "| " + row["line"] + " | " + row["method"] + " | " + " | ".join(f"{float(row[k]):.6f}" for k in METRICS) + " |\n"
    text += "\nBoth primary macro contrasts favor old predictor-error supervision over this biological-magnitude supervision control. The estimate is conditional on the same prediction/Public inputs, learner and Source family. It does not remove prediction information, establish value beyond every biological proxy, identify a root cause, or establish external transfer. Secondary metrics are retained; the control has lower macro error_at_10 in both directions, so the advantage is not universal across endpoints.\n\n"
    text += "The Source population is the existing 1808 biological tasks / 575 gene clusters. Both directions share biological truth. The 14464 norm-supervised training rows are repeated fit-row instances across ten fits, not independent new experiments. The new control uses zero realized predictor-error training labels; the ten reused original models retain their historical error-label training. Cached Source prediction inputs remain in both arms; only new upstream calls are zero.\n\n"
    text += "Counts are the exact saved 5000 × 575 Source gene multiplicities, generation seed 20261002; this is not a reproduction of the original main bootstrap seed. All contexts, both contrast signs and all seven metrics remain in the saved CSVs. Nominal paired intervals describe Source DEV/SEEN sampling conditional on these fitted models; no fresh confirmation, formal-method promotion or old external champion selection is claimed.\n\n"
    text += "Runtime predictions, models and draw arrays stay server-side in `" + str(O) + "`. The immutable original execution cost was 16.752635 seconds wall, 16.740402 seconds CPU, peak RSS 566534144 bytes; ten new fits and ten model reuses. The original terminal code is 1; this report-only recovery terminal code is 0.\n"
    with summary.open("x") as f: f.write(text)
    after = {path: sha(path) for path in before}
    if before != after: raise RuntimeError("An original saved input changed during rendering")
    result = {"status": "COMPLETE_SAVED_STATISTICS_RENDER_ONLY_RECOVERY", "original_terminal_code": 1,
        "recovery_terminal_code": 0, "original_status_preserved": "FAILED_optional_tabulate_formatting",
        "new_fits": 0, "new_scores": 0, "new_statistics": 0, "new_RNG": 0,
        "authorized_control_new_fits_already_completed": 10, "reused_original_rank_models": 10,
        "distinct_Source_biological_tasks": 1808, "distinct_Source_gene_clusters": 575,
        "biological_norm_fit_row_instances": 14464, "control_realized_error_training_labels": 0,
        "retained_cached_Source_prediction_inputs": True, "formal_core_unchanged": True,
        "input_sha256_before": before, "input_sha256_after": after,
        "renderer_sha256": sha(Path(__file__)), "summary_sha256": sha(summary), "runtime": str(O)}
    with receipt.open("x") as f: f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "terminal_code": 0, "summary": str(summary), "receipt": str(receipt)}))


if __name__ == "__main__": main()

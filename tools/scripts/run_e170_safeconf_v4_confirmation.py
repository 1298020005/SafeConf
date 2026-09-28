#!/usr/bin/env python3
"""One-shot SafeConf-v4 confirmation on the four outcome-sealed E170 panels."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import run_safeconf_v4_development as core  # noqa: E402


STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
PUBLIC = Path("/home/yyf/proj/docs/实验结果/E170_primary_cd4_multipanel_precision_20260718")
F2_ROOT = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/E170")
TRUTH_ROOT = Path("/home/yyf/data/safeconf_v4_e170_confirmation_truth")
OUT = STAGE / "confirmation/e170_primary_cd4_four_panel"
PANELS = ("P01", "P02", "P03", "P04")
SEED = 20260930
N_BOOTSTRAP = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key], np.float64) for key in archive.files}


def verify_authorization() -> dict:
    auth_path = STAGE / "E170_V4_CONFIRMATION_AUTHORIZATION.json"
    auth = json.loads(auth_path.read_text())
    if auth.get("status") != "AUTHORIZED_AFTER_FINAL_CANDIDATE_FREEZE":
        raise RuntimeError("E170 v4 confirmation is not authorized")
    if auth.get("all_four_panels_required") is not True or auth.get("test_truth_opened_at_authorization") is not False:
        raise RuntimeError("E170 authorization panel/truth contract failed")
    for rel, expected in auth["frozen_file_sha256"].items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen confirmation file changed: {rel}")
    competence = pd.read_csv(STAGE / "UPSTREAM_COMPETENCE_V4.csv")
    row = competence.loc[
        competence.asset.eq("E170_primary_CD4_four_panel")
        & competence.upstream_model.eq("scGPT_GEARS_6member_ensemble")
    ]
    if len(row) != 1 or not bool(row.iloc[0].passes_2pct_competence_gate):
        raise RuntimeError("E170 upstream competence gate is absent or failed")
    return auth


def universal_features(prediction: np.ndarray) -> dict[str, np.ndarray]:
    absolute = np.abs(prediction)
    q95 = np.quantile(absolute, 0.95, axis=1)
    scale = np.maximum(q95, 1e-12)
    return {
        "predicted_magnitude": np.sqrt(np.mean(prediction**2, axis=1)),
        "prediction_abs_mean": absolute.mean(axis=1),
        "prediction_signed_mean": prediction.mean(axis=1),
        "prediction_std": prediction.std(axis=1),
        "prediction_abs_q95": q95,
        "prediction_sparsity": np.mean(absolute <= scale[:, None] * 0.01, axis=1),
    }


def history_maps(panel: str, interface: pd.DataFrame, effects: dict[str, np.ndarray]) -> dict:
    train = interface[interface.split.eq("train")].copy()
    guide_index = pd.read_csv(F2_ROOT / panel / "F2_pretruth/PRETRUTH_GUIDE_EFFECT_INDEX.csv", keep_default_na=False)
    source: dict[tuple[str, str], dict] = {}
    for (gene, condition), group in train.groupby(["perturbed_gene_id", "culture_condition"], sort=True):
        vectors = np.stack([effects[key] for key in group.effect_asset_key.astype(str)])
        mean = vectors.mean(axis=0)
        dispersion = float(np.sqrt(np.mean((vectors - mean) ** 2)))
        task_ids = set(group.task_id.astype(str))
        guides = guide_index[guide_index.task_id.astype(str).isin(task_ids)]
        source[(str(gene), str(condition))] = {
            "effect": mean,
            "dispersion": dispersion,
            "n_source_cells": float(guides.source_rows_merged_before_normalization.sum()),
            "n_source_contexts": float(group.donor_id.nunique()),
            "n_source_batches": float(guides.guide_id.nunique()),
            "min_source_cells": float(guides.source_rows_merged_before_normalization.min()),
        }
    return source


def feature_frame(panel: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    release = PUBLIC / "pretruth_release" / panel
    interface = pd.read_csv(release / "tables/PRETRUTH_SCORING_INTERFACE.csv", keep_default_na=False)
    interface["prediction_row"] = np.arange(len(interface))
    with np.load(release / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False) as archive:
        prediction = np.asarray(archive["ensemble_seed_family_mean"], np.float64)
    if prediction.shape != (len(interface), 512):
        raise RuntimeError(f"{panel} prediction/interface contract changed")
    effects = load_npz(F2_ROOT / panel / "F2_pretruth/SEEN_TARGET_EFFECTS.npz")
    source = history_maps(panel, interface, effects)

    def build(rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy().reset_index(drop=True)
        pred = prediction[rows.prediction_row.to_numpy(int)]
        frame = rows[[
            "task_id", "panel_id", "donor_id", "culture_condition",
            "perturbed_gene_id", "perturbed_gene_name", "target_stratum", "split",
        ]].copy()
        frame["gene"] = panel + "::" + frame.perturbed_gene_id.astype(str)
        frame["target"] = frame.panel_id.astype(str) + "::" + frame.culture_condition.astype(str)
        for name, values in universal_features(pred).items():
            frame[name] = values

        support_rows = []
        for gene, condition, vector in zip(
            frame.perturbed_gene_id.astype(str), frame.culture_condition.astype(str), pred
        ):
            history = source.get((gene, condition))
            if history is None:
                support_rows.append({
                    "n_source_cells": 0.0, "n_source_contexts": 0.0,
                    "n_source_batches": 0.0, "min_source_cells": 0.0,
                    "prediction_source_cosine": np.nan,
                    "negative_model_source_gap": np.nan,
                    "source_delta_dispersion": np.nan,
                    "conflict_missing": 1.0,
                    "source_transfer_magnitude": np.nan,
                })
                continue
            effect = history["effect"]
            denominator = float(np.linalg.norm(vector) * np.linalg.norm(effect))
            cosine = float(np.dot(vector, effect) / denominator) if denominator > 1e-12 else 0.0
            gap = float(np.sqrt(np.mean((vector - effect) ** 2)))
            support_rows.append({
                "n_source_cells": history["n_source_cells"],
                "n_source_contexts": history["n_source_contexts"],
                "n_source_batches": history["n_source_batches"],
                "min_source_cells": history["min_source_cells"],
                "prediction_source_cosine": cosine,
                "negative_model_source_gap": -gap,
                "source_delta_dispersion": history["dispersion"],
                "conflict_missing": 0.0,
                "source_transfer_magnitude": float(np.sqrt(np.mean(effect**2))),
            })
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(support_rows)], axis=1)

    validation = build(interface[interface.split.eq("validation")])
    test = build(interface[interface.primary_test_task.astype(bool)])
    if len(validation) != 480 or len(test) != 600:
        raise RuntimeError(f"{panel} validation/test task count changed")
    validation["true_error_rmse"] = [
        float(np.sqrt(np.mean((prediction[row.prediction_row] - effects[row.effect_asset_key]) ** 2)))
        for row in interface[interface.split.eq("validation")].itertuples()
    ]
    return validation, test, {row.task_id: prediction[row.prediction_row] for row in interface[interface.primary_test_task.astype(bool)].itertuples()}


def fit_confirmation(fit: pd.DataFrame, query: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scores = {
        "Magnitude_raw": query.predicted_magnitude.to_numpy(float),
        "Ridge_U": core.fit_model(fit, query, core.GROUPS["U"], "Ridge", SEED),
        "Ridge_US": core.fit_model(fit, query, core.GROUPS["US"], "Ridge", SEED + 1),
        "Ridge_USR": core.fit_model(fit, query, core.GROUPS["USR"], "Ridge", SEED + 2),
        "Ridge_USRCH": core.fit_model(fit, query, core.GROUPS["USRCH"], "Ridge", SEED + 3),
    }
    v2, gate = core.nested_v2(fit, query, SEED + 1000)
    scores["V2_nested"] = v2
    rows = []
    for method, values in scores.items():
        for task, panel, state, stratum, gene, value in zip(
            query.task_id.astype(str), query.panel_id.astype(str), query.culture_condition.astype(str),
            query.target_stratum.astype(str), query.gene.astype(str), values,
        ):
            rows.append({
                "task_id": task, "panel_id": panel, "culture_condition": state,
                "target_stratum": stratum, "gene": gene, "method": method,
                "predicted_risk": float(value),
            })
    return pd.DataFrame(rows), gate


def load_test_truth(prediction_vectors: dict[str, np.ndarray]) -> dict[str, float]:
    errors = {}
    for panel in PANELS:
        truth = load_npz(TRUTH_ROOT / panel / "TEST_TARGET_EFFECTS.npz")
        expected = {task for task in prediction_vectors if task.startswith(f"E170::{panel}::")}
        if set(truth) != expected or len(truth) != 600:
            raise RuntimeError(f"{panel} truth/prediction task identity mismatch")
        for task in sorted(expected):
            errors[task] = float(np.sqrt(np.mean((prediction_vectors[task] - truth[task]) ** 2)))
    return errors


def evaluate(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = []
    for (panel, state, method), group in predictions.groupby(["panel_id", "culture_condition", "method"], sort=True):
        ids = group.task_id.to_numpy(str)
        risk = group.predicted_risk.to_numpy(float)
        truth = group.true_error_rmse.to_numpy(float)
        rows.append({
            "panel_id": panel, "culture_condition": state, "method": method,
            "n_tasks": len(group), "utility20": core.utility20(ids, risk, truth),
            "spearman": core.rho(risk, truth), **core.selective_metrics(ids, risk, truth),
        })
        order = np.lexsort((ids, risk))
        for fraction in (0.1, 0.2, 0.5, 1.0):
            k = max(1, int(math.ceil(fraction * len(group))))
            curves.append({
                "panel_id": panel, "culture_condition": state, "method": method,
                "coverage": fraction, "n_accepted": k,
                "mean_true_rmse": float(truth[order[:k]].mean()),
            })
    strata = pd.DataFrame(rows)
    summary = strata.groupby("method", as_index=False).agg(
        n_strata=("utility20", "count"), utility20=("utility20", "mean"),
        spearman=("spearman", "mean"), aurc=("aurc", "mean"),
        risk_at_10=("risk_at_10", "mean"), risk_at_20=("risk_at_20", "mean"),
        risk_at_50=("risk_at_50", "mean"), high_risk_miss_rate=("high_risk_miss_rate", "mean"),
    ).sort_values(["utility20", "spearman"], ascending=False)
    return strata, summary, pd.DataFrame(curves)


def subgroup_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stratum, method), group in predictions.groupby(["target_stratum", "method"], sort=True):
        ids = group.task_id.to_numpy(str)
        risk = group.predicted_risk.to_numpy(float)
        truth = group.true_error_rmse.to_numpy(float)
        rows.append({
            "target_stratum": stratum, "method": method, "n_tasks": len(group),
            "utility20": core.utility20(ids, risk, truth), "spearman": core.rho(risk, truth),
            **core.selective_metrics(ids, risk, truth),
        })
    return pd.DataFrame(rows)


def bootstrap(predictions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("FinalV2_vs_Magnitude", "V2_nested", "Magnitude_raw"),
        ("V1_vs_Magnitude_secondary", "Ridge_USR", "Magnitude_raw"),
        ("FinalV2_vs_V1_secondary", "V2_nested", "Ridge_USR"),
        ("V1_vs_UniversalP_secondary", "Ridge_USR", "Ridge_U"),
    ]
    wide = predictions.pivot(
        index=["task_id", "panel_id", "culture_condition", "gene", "true_error_rmse"],
        columns="method", values="predicted_risk",
    ).reset_index()
    if len(wide) != 2400 or wide.isna().any().any():
        raise RuntimeError("confirmation OOF matrix is incomplete")
    ids = wide.task_id.to_numpy(str)
    truth = wide.true_error_rmse.to_numpy(float)
    panels = wide.panel_id.to_numpy(str)
    states = wide.culture_condition.to_numpy(str)
    risks = {name: wide[name].to_numpy(float) for _, a, b in comparisons for name in (a, b)}
    panel_clusters = {}
    for panel in PANELS:
        genes = sorted(wide.loc[wide.panel_id.eq(panel), "gene"].unique())
        panel_clusters[panel] = [np.flatnonzero(wide.gene.to_numpy(str) == gene) for gene in genes]
        if len(genes) != 200:
            raise RuntimeError(f"{panel} does not contain 200 target clusters")
    rng = np.random.default_rng(SEED)
    u_draws = {name: [] for name, _, _ in comparisons}
    rho_draws = {name: [] for name, _, _ in comparisons}
    for _ in range(N_BOOTSTRAP):
        selected_rows = []
        for panel in PANELS:
            clusters = panel_clusters[panel]
            selected_rows.extend(clusters[index] for index in rng.integers(0, len(clusters), len(clusters)))
        take = np.concatenate(selected_rows)
        values = {}
        for method, risk in risks.items():
            us, rs = [], []
            for panel in PANELS:
                for state in sorted(set(states)):
                    use = take[(panels[take] == panel) & (states[take] == state)]
                    us.append(core.utility20(ids[use], risk[use], truth[use]))
                    rs.append(core.rho(risk[use], truth[use]))
            values[method] = (float(np.nanmean(us)), float(np.nanmean(rs)))
        for name, left, right in comparisons:
            u_draws[name].append(values[left][0] - values[right][0])
            rho_draws[name].append(values[left][1] - values[right][1])
    point = summary.set_index("method")
    rows = []
    for name, left, right in comparisons:
        u = np.asarray(u_draws[name])
        r = np.asarray(rho_draws[name])
        rows.append({
            "comparison": name, "method_a": left, "method_b": right,
            "delta_utility20": float(point.loc[left, "utility20"] - point.loc[right, "utility20"]),
            "utility_ci95_lower": float(np.nanquantile(u, 0.025)),
            "utility_ci95_upper": float(np.nanquantile(u, 0.975)),
            "delta_spearman": float(point.loc[left, "spearman"] - point.loc[right, "spearman"]),
            "spearman_ci95_lower": float(np.nanquantile(r, 0.025)),
            "spearman_ci95_upper": float(np.nanquantile(r, 0.975)),
            "bootstrap_replicates": N_BOOTSTRAP,
        })
    return pd.DataFrame(rows)


def gate_a(strata: pd.DataFrame, summary: pd.DataFrame) -> dict:
    point = summary.set_index("method")
    wide = strata.pivot(index=["panel_id", "culture_condition"], columns="method", values="utility20")
    delta = float(point.loc["V2_nested", "utility20"] - point.loc["Magnitude_raw", "utility20"])
    def degradation(column: str) -> float:
        base = float(point.loc["Magnitude_raw", column])
        final = float(point.loc["V2_nested", column])
        return max(0.0, (final - base) / max(abs(base), 1e-12))
    result = {
        "delta_u20_macro": delta,
        "nonnegative_strata_fraction": float(((wide.V2_nested - wide.Magnitude_raw) >= 0).mean()),
        "risk10_relative_degradation": degradation("risk_at_10"),
        "risk20_relative_degradation": degradation("risk_at_20"),
        "risk50_relative_degradation": degradation("risk_at_50"),
        "high_risk_miss_rate_degradation": max(0.0, float(
            point.loc["V2_nested", "high_risk_miss_rate"] - point.loc["Magnitude_raw", "high_risk_miss_rate"]
        )),
        "aurc_relative_degradation": degradation("aurc"),
        "valid_strata_fraction": float(wide[["V2_nested", "Magnitude_raw"]].notna().all(axis=1).mean()),
        "n_planned_strata": 12,
    }
    result["gate_a_pass"] = bool(
        result["delta_u20_macro"] >= 0.005
        and result["nonnegative_strata_fraction"] >= 0.60
        and max(result["risk10_relative_degradation"], result["risk20_relative_degradation"], result["risk50_relative_degradation"]) <= 0.05
        and result["high_risk_miss_rate_degradation"] <= 0.02
        and result["aurc_relative_degradation"] <= 0.05
        and result["valid_strata_fraction"] >= 0.80
    )
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite one-shot confirmation: {OUT}")
    auth = verify_authorization()
    started = time.time()
    fit_parts, test_parts, prediction_vectors = [], [], {}
    for panel in PANELS:
        fit, test, vectors = feature_frame(panel)
        fit_parts.append(fit)
        test_parts.append(test)
        prediction_vectors.update(vectors)
    fit = pd.concat(fit_parts, ignore_index=True)
    query = pd.concat(test_parts, ignore_index=True)
    if len(fit) != 1920 or len(query) != 2400 or fit.gene.nunique() != 640 or query.gene.nunique() != 800:
        raise RuntimeError("E170 confirmation population contract changed")

    # Generate every frozen risk score before reading confirmation truth.
    predictions, gate_fit = fit_confirmation(fit, query)
    errors = load_test_truth(prediction_vectors)
    predictions["true_error_rmse"] = predictions.task_id.map(errors).astype(float)
    strata, summary, curves = evaluate(predictions)
    subgroups = subgroup_results(predictions)
    paired = bootstrap(predictions, summary)
    gate = gate_a(strata, summary)

    OUT.mkdir(parents=True)
    query.to_csv(OUT / "CONFIRMATION_FEATURES.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    predictions.to_csv(OUT / "CONFIRMATION_PREDICTIONS.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    strata.to_csv(OUT / "STRATUM_RESULTS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    curves.to_csv(OUT / "RISK_COVERAGE.csv", index=False)
    subgroups.to_csv(OUT / "HISTORY_AVAILABILITY_RESULTS.csv", index=False)
    paired.to_csv(OUT / "PAIRED_CLUSTER_BOOTSTRAP.csv", index=False)
    pd.DataFrame([gate_fit]).to_csv(OUT / "V2_GATE_FIT.csv", index=False)
    (OUT / "GATE_A_RESULT.json").write_text(json.dumps(gate, indent=2) + "\n")

    report = [
        "# E170 SafeConf-v4 one-shot confirmation",
        "",
        "All four outcome-sealed panels were opened together after Final Candidate freeze.",
        "",
        "| Method | Utility@20 | Spearman | AURC | risk@20 | miss rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        report.append(
            f"| {row.method} | {row.utility20:.6f} | {row.spearman:.6f} | {row.aurc:.6f} | "
            f"{row.risk_at_20:.6f} | {row.high_risk_miss_rate:.6f} |"
        )
    report.extend(["", "## Gate A", "", "```json", json.dumps(gate, indent=2), "```", "", "## Paired bootstrap", ""])
    for row in paired.itertuples():
        report.append(
            f"- {row.comparison}: ΔU20={row.delta_utility20:+.6f} "
            f"[{row.utility_ci95_lower:+.6f}, {row.utility_ci95_upper:+.6f}], "
            f"Δρ={row.delta_spearman:+.6f} "
            f"[{row.spearman_ci95_lower:+.6f}, {row.spearman_ci95_upper:+.6f}]."
        )
    report.extend([
        "", "## Evidence boundary", "",
        "- This confirms or rejects the frozen V2 on new perturbations and one held-out donor within the same study.",
        "- It is not an external-study confirmation.",
        "- Frozen V1 remains a secondary benchmark and cannot replace V2 after seeing this result.",
        "- Column-unseen tasks are retained and reported; no-history failures are not removed.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    status = {
        "status": "COMPLETE",
        "one_shot_confirmation": True,
        "all_four_panels_opened_together": True,
        "n_fit_validation_tasks": int(len(fit)),
        "n_confirmation_tasks": int(len(query)),
        "n_target_clusters": int(query.gene.nunique()),
        "n_evaluation_strata": int(len(strata) / summary.method.nunique()),
        "final_candidate": "V2_nested_evidence_shrinkage",
        "final_candidate_changed": False,
        "gate_a": gate,
        "gate_b": "NOT_EVALUABLE_NO_FROZEN_QUALITY_FIELDS",
        "authorization_sha256": sha256(STAGE / "E170_V4_CONFIRMATION_AUTHORIZATION.json"),
        "truth_manifest_sha256": sha256(TRUTH_ROOT / "MANIFEST.sha256"),
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()

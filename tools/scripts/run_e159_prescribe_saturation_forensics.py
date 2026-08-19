#!/usr/bin/env python3
"""E159: post-unseal forensics for PRESCRIBE score saturation.

This runner is deliberately restricted to already-materialized E157/E158 tabular
artifacts and a fixed set of PRESCRIBE source files.  It must not open an h5ad
container or materialize any expression matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EXPERIMENT = "E159_prescribe_saturation_forensics"
ANALYSIS_TIMING = "post_unseal_forensic_not_preregistered"
PANELS = ("Norman_P3", "Norman_P4")
SCORE_COLUMNS = (
    "log_prob",
    "epistemic_confidence",
    "aleatoric_confidence",
    "combined_confidence_official",
    "predicted_magnitude_rms",
)
PCA_COLUMNS = tuple(f"predicted_pca_{idx}" for idx in range(10))
PRIMARY_ENDPOINTS = (
    ("pearson_effect_accuracy", "PCA主accuracy", "positive"),
    ("frac_correct_direction_all", "PCA主direction", "positive"),
    ("rmse_effect_error", "PCA主RMSE", "negative"),
)
RAW_SENSITIVITY_ENDPOINTS = (
    (
        "raw_pearson_effect_accuracy_sensitivity",
        "raw sensitivity: Pearson accuracy",
        "positive",
    ),
    (
        "raw_frac_correct_direction_all_sensitivity",
        "raw sensitivity: direction",
        "positive",
    ),
    ("raw_rmse_effect_error_sensitivity", "raw sensitivity: RMSE", "negative"),
)

# Frozen E157 native configuration.  output_dim is independently checked from
# the ten locked predicted_pca_* columns.  The fixed PRESCRIBE source is also
# checked for the exp-budget and +/-bound implementation.
CERTAINTY_BUDGET = "exp"
OUTPUT_DIM = 10
LOG_SCALE = 10.0
EVIDENCE_BOUND = 30.0
TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")


@dataclass(frozen=True)
class InputSpec:
    role: str
    path: Path
    display_path: str


def build_input_specs(repo: Path, prescribe: Path) -> list[InputSpec]:
    e157 = repo / "docs/实验结果/E157_prescribe_norman_p3p4_native_20260714"
    e158 = repo / "docs/实验结果/E158_prescribe_norman_p3p4_forward_evaluation_20260714/attempt_001"
    return [
        InputSpec(
            "E157_locked_scores_P3",
            e157
            / "norman_p3_formal_seed3407/locked/E157_LOCKED_LABEL_ONLY_TASK_SCORES.csv",
            "docs/实验结果/E157_prescribe_norman_p3p4_native_20260714/"
            "norman_p3_formal_seed3407/locked/E157_LOCKED_LABEL_ONLY_TASK_SCORES.csv",
        ),
        InputSpec(
            "E157_locked_scores_P4",
            e157
            / "norman_p4_formal_seed3407/locked/E157_LOCKED_LABEL_ONLY_TASK_SCORES.csv",
            "docs/实验结果/E157_prescribe_norman_p3p4_native_20260714/"
            "norman_p4_formal_seed3407/locked/E157_LOCKED_LABEL_ONLY_TASK_SCORES.csv",
        ),
        InputSpec(
            "E158_materialized_task_metrics",
            e158 / "tables/E158_TASK_METRICS.csv",
            "docs/实验结果/E158_prescribe_norman_p3p4_forward_evaluation_20260714/"
            "attempt_001/tables/E158_TASK_METRICS.csv",
        ),
        InputSpec(
            "E158_attempt_status",
            e158 / "RUN_STATUS.json",
            "docs/实验结果/E158_prescribe_norman_p3p4_forward_evaluation_20260714/"
            "attempt_001/RUN_STATUS.json",
        ),
        InputSpec(
            "E158_unseal_event",
            e158 / "UNSEAL_EVENT.json",
            "docs/实验结果/E158_prescribe_norman_p3p4_forward_evaluation_20260714/"
            "attempt_001/UNSEAL_EVENT.json",
        ),
        InputSpec(
            "PRESCRIBE_evidence_scaler_source",
            prescribe / "src/nn/scaler.py",
            "PRESCRIBE/src/nn/scaler.py",
        ),
        InputSpec(
            "PRESCRIBE_model_source",
            prescribe / "src/nn/model.py",
            "PRESCRIBE/src/nn/model.py",
        ),
        InputSpec(
            "PRESCRIBE_posterior_update_source",
            prescribe / "src/distributions/multinormal.py",
            "PRESCRIBE/src/distributions/multinormal.py",
        ),
        InputSpec(
            "PRESCRIBE_confidence_source",
            prescribe / "src/model/lightening_module.py",
            "PRESCRIBE/src/model/lightening_module.py",
        ),
        InputSpec(
            "PRESCRIBE_encoder_source",
            prescribe / "src/nn/encoder/encoder.py",
            "PRESCRIBE/src/nn/encoder/encoder.py",
        ),
    ]


def load_allowlisted_inputs(
    specs: list[InputSpec],
) -> tuple[dict[str, bytes], pd.DataFrame]:
    payloads: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.path.is_file():
            raise FileNotFoundError(f"Missing allowlisted input: {spec.path}")
        if spec.path.suffix.lower() in {".h5ad", ".h5", ".loom", ".zarr"}:
            raise RuntimeError(f"Expression container is forbidden in E159: {spec.path}")
        payload = spec.path.read_bytes()
        payloads[spec.role] = payload
        rows.append(
            {
                "role": spec.role,
                "display_path": spec.display_path,
                "absolute_path": str(spec.path),
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "read_scope": "allowlisted_existing_artifact_or_fixed_source",
            }
        )
    return payloads, pd.DataFrame(rows)


def csv_from_payload(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload))


def json_from_payload(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


def assert_fixed_source_contract(payloads: dict[str, bytes]) -> None:
    required_tokens = {
        "PRESCRIBE_evidence_scaler_source": (
            'elif budget == "exp"',
            "self.log_scale = dim",
            "lower=-self.bound, upper=self.bound",
        ),
        "PRESCRIBE_model_source": (
            "log_evidence = self.scaler.forward(log_prob)",
            "return D.PosteriorUpdate(sufficient_statistics, log_evidence), log_prob",
        ),
        "PRESCRIBE_posterior_update_source": (
            "evidence = update.log_evidence.exp()",
            "evidence_proportion = 1 - prior_proportion",
        ),
        "PRESCRIBE_confidence_source": (
            "epistemic_conf = 10 * evidence_proportion + 10",
            "aleatoric_conf = -y_pred.maximum_a_posteriori().entropy()",
        ),
        "PRESCRIBE_encoder_source": (
            "pert_emb = self.f11",
            "new_pert_idx.append([self.pert2id[g] for g in p])",
        ),
    }
    for role, tokens in required_tokens.items():
        source = payloads[role].decode("utf-8")
        missing = [token for token in tokens if token not in source]
        if missing:
            raise RuntimeError(f"Fixed-source contract mismatch for {role}: {missing}")


def validate_and_join(
    p3: pd.DataFrame, p4: pd.DataFrame, metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    locked = pd.concat([p3, p4], ignore_index=True)
    needed_locked = {"panel", "task_id", *SCORE_COLUMNS, *PCA_COLUMNS}
    needed_metrics = {
        "panel",
        "task_id",
        "n_test_cells",
        *(name for name, _, _ in PRIMARY_ENDPOINTS),
        *(name for name, _, _ in RAW_SENSITIVITY_ENDPOINTS),
    }
    if missing := sorted(needed_locked - set(locked.columns)):
        raise RuntimeError(f"Locked score columns missing: {missing}")
    if missing := sorted(needed_metrics - set(metrics.columns)):
        raise RuntimeError(f"E158 metric columns missing: {missing}")
    if len(PCA_COLUMNS) != OUTPUT_DIM:
        raise RuntimeError("Frozen output dimension does not match the PCA score contract")
    if locked.duplicated(["panel", "task_id"]).any():
        raise RuntimeError("Duplicate panel/task_id in E157 locked scores")
    if metrics.duplicated(["panel", "task_id"]).any():
        raise RuntimeError("Duplicate panel/task_id in E158 task metrics")
    for panel in PANELS:
        if int((locked["panel"] == panel).sum()) != 24:
            raise RuntimeError(f"Expected 24 locked tasks for {panel}")
        if int((metrics["panel"] == panel).sum()) != 24:
            raise RuntimeError(f"Expected 24 E158 task rows for {panel}")

    metric_columns = [
        "panel",
        "task_id",
        "n_test_cells",
        *(name for name, _, _ in PRIMARY_ENDPOINTS),
        *(name for name, _, _ in RAW_SENSITIVITY_ENDPOINTS),
    ]
    joined = locked.merge(
        metrics[metric_columns],
        on=["panel", "task_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("E157/E158 task identity mismatch")
    joined = joined.drop(columns="_merge").sort_values(
        ["panel", "task_id"], ignore_index=True
    )

    audit_rows: list[dict[str, Any]] = []
    for panel in PANELS:
        left = locked.loc[locked["panel"] == panel].sort_values("task_id")
        right = metrics.loc[metrics["panel"] == panel].sort_values("task_id")
        for column in (
            "epistemic_confidence",
            "aleatoric_confidence",
            "combined_confidence_official",
            "predicted_magnitude_rms",
        ):
            delta = np.abs(
                left[column].to_numpy(dtype=float) - right[column].to_numpy(dtype=float)
            )
            audit_rows.append(
                {
                    "panel": panel,
                    "field": column,
                    "n_rows": len(delta),
                    "max_abs_E157_vs_E158_delta": float(delta.max()),
                    "within_tolerance_1e-7": bool(np.all(delta <= 1e-7)),
                }
            )
    audit = pd.DataFrame(audit_rows)
    if not audit["within_tolerance_1e-7"].all():
        raise RuntimeError("E157 locked score and E158 carried score mismatch")
    return joined, audit


def summarize_scores(locked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        subset = locked.loc[locked["panel"] == panel]
        for score in SCORE_COLUMNS:
            values = subset[score].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            nunique = int(pd.Series(finite).nunique(dropna=True))
            rows.append(
                {
                    "panel": panel,
                    "score": score,
                    "n": len(values),
                    "n_finite": len(finite),
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "std_population_ddof0": float(finite.std(ddof=0)),
                    "nunique": nunique,
                    "constant": nunique < 2,
                    "estimable_for_within_panel_rank_association": nunique >= 2,
                }
            )
    return pd.DataFrame(rows)


def summarize_pca(locked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        subset = locked.loc[locked["panel"] == panel]
        for score in PCA_COLUMNS:
            values = subset[score].to_numpy(dtype=float)
            rows.append(
                {
                    "panel": panel,
                    "pca_coordinate": score,
                    "n": len(values),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "std_population_ddof0": float(values.std(ddof=0)),
                    "nunique": int(pd.Series(values).nunique()),
                    "constant": int(pd.Series(values).nunique()) < 2,
                }
            )
    return pd.DataFrame(rows)


def reconstruct_scaler_chain(locked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for row in locked.itertuples(index=False):
        raw = np.float32(row.log_prob)
        pre_clamp = np.float32(raw + np.float32(LOG_SCALE))
        clamped = np.float32(np.clip(pre_clamp, -EVIDENCE_BOUND, EVIDENCE_BOUND))
        evidence = np.float32(np.exp(clamped))
        denominator = np.float32(np.float32(1.0) + evidence)
        prior_proportion = np.float32(np.float32(1.0) / denominator)
        unstable = np.float32(np.float32(1.0) - prior_proportion)
        stable = np.float32(evidence / denominator)
        epistemic_reconstructed = np.float32(
            np.float32(10.0) * unstable + np.float32(10.0)
        )
        rows.append(
            {
                "panel": row.panel,
                "task_id": row.task_id,
                "raw_log_prob_float32": float(raw),
                "certainty_budget": CERTAINTY_BUDGET,
                "output_dim": OUTPUT_DIM,
                "log_scale": LOG_SCALE,
                "pre_clamp_log_evidence_float32": float(pre_clamp),
                "lower_bound": -EVIDENCE_BOUND,
                "upper_bound": EVIDENCE_BOUND,
                "clamp_active": bool(pre_clamp < -EVIDENCE_BOUND or pre_clamp > EVIDENCE_BOUND),
                "clamped_log_evidence_float32": float(clamped),
                "evidence_exp_float32": float(evidence),
                "prior_proportion_float32": float(prior_proportion),
                "evidence_proportion_unstable_1_minus_prior_float32": float(unstable),
                "evidence_proportion_stable_e_over_1_plus_e_float32": float(stable),
                "epistemic_reconstructed_from_unstable_float32": float(
                    epistemic_reconstructed
                ),
                "epistemic_locked": float(row.epistemic_confidence),
            }
        )
    tasks = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for panel in PANELS:
        subset = tasks.loc[tasks["panel"] == panel]
        summary_rows.append(
            {
                "panel": panel,
                "n_tasks": len(subset),
                "raw_log_prob_min": subset["raw_log_prob_float32"].min(),
                "raw_log_prob_max": subset["raw_log_prob_float32"].max(),
                "pre_clamp_min": subset["pre_clamp_log_evidence_float32"].min(),
                "pre_clamp_max": subset["pre_clamp_log_evidence_float32"].max(),
                "clamp_fraction": subset["clamp_active"].mean(),
                "clamped_log_evidence_nunique": subset[
                    "clamped_log_evidence_float32"
                ].nunique(),
                "evidence_exp_float32_min": subset["evidence_exp_float32"].min(),
                "evidence_exp_float32_max": subset["evidence_exp_float32"].max(),
                "unstable_evidence_proportion_nunique": subset[
                    "evidence_proportion_unstable_1_minus_prior_float32"
                ].nunique(),
                "unstable_evidence_proportion_min": subset[
                    "evidence_proportion_unstable_1_minus_prior_float32"
                ].min(),
                "unstable_evidence_proportion_max": subset[
                    "evidence_proportion_unstable_1_minus_prior_float32"
                ].max(),
                "stable_evidence_proportion_min": subset[
                    "evidence_proportion_stable_e_over_1_plus_e_float32"
                ].min(),
                "stable_evidence_proportion_max": subset[
                    "evidence_proportion_stable_e_over_1_plus_e_float32"
                ].max(),
                "epistemic_reconstructed_nunique": subset[
                    "epistemic_reconstructed_from_unstable_float32"
                ].nunique(),
            }
        )
    return tasks, pd.DataFrame(summary_rows)


def safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    x_unique = int(pair["x"].nunique())
    y_unique = int(pair["y"].nunique())
    if n < 3:
        return {
            "n_tasks": n,
            "x_nunique": x_unique,
            "y_nunique": y_unique,
            "rho": math.nan,
            "pvalue_two_sided": math.nan,
            "status": "undefined_insufficient_pairs",
        }
    if x_unique < 2:
        return {
            "n_tasks": n,
            "x_nunique": x_unique,
            "y_nunique": y_unique,
            "rho": math.nan,
            "pvalue_two_sided": math.nan,
            "status": "undefined_constant_score",
        }
    if y_unique < 2:
        return {
            "n_tasks": n,
            "x_nunique": x_unique,
            "y_nunique": y_unique,
            "rho": math.nan,
            "pvalue_two_sided": math.nan,
            "status": "undefined_constant_endpoint",
        }
    result = spearmanr(pair["x"].to_numpy(), pair["y"].to_numpy())
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    if not np.isfinite(rho):
        return {
            "n_tasks": n,
            "x_nunique": x_unique,
            "y_nunique": y_unique,
            "rho": math.nan,
            "pvalue_two_sided": math.nan,
            "status": "undefined_scipy_nonfinite",
        }
    return {
        "n_tasks": n,
        "x_nunique": x_unique,
        "y_nunique": y_unique,
        "rho": rho,
        "pvalue_two_sided": pvalue,
        "status": "estimable_posthoc",
    }


def posthoc_spearman(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoint_specs = (*PRIMARY_ENDPOINTS, *RAW_SENSITIVITY_ENDPOINTS)
    panel_rhos: dict[str, list[float]] = {name: [] for name, _, _ in endpoint_specs}
    for endpoint, label, expected_direction in endpoint_specs:
        for panel in PANELS:
            subset = joined.loc[joined["panel"] == panel]
            result = safe_spearman(subset["log_prob"], subset[endpoint])
            rows.append(
                {
                    "analysis_timing": ANALYSIS_TIMING,
                    "scope": panel,
                    "score": "raw_log_prob",
                    "endpoint": endpoint,
                    "endpoint_label": label,
                    "expected_direction": expected_direction,
                    **result,
                    "interpretation_constraint": "hypothesis_generating_only",
                }
            )
            if np.isfinite(result["rho"]):
                panel_rhos[endpoint].append(float(result["rho"]))

        pooled = safe_spearman(joined["log_prob"], joined[endpoint])
        rows.append(
            {
                "analysis_timing": ANALYSIS_TIMING,
                "scope": "pooled_48_diagnostic",
                "score": "raw_log_prob",
                "endpoint": endpoint,
                "endpoint_label": label,
                "expected_direction": expected_direction,
                **pooled,
                "interpretation_constraint": "pooled_panels_not_independent_hypothesis_generating_only",
            }
        )
        rhos = panel_rhos[endpoint]
        rows.append(
            {
                "analysis_timing": ANALYSIS_TIMING,
                "scope": "equal_panel_macro",
                "score": "raw_log_prob",
                "endpoint": endpoint,
                "endpoint_label": label,
                "expected_direction": expected_direction,
                "n_tasks": len(joined),
                "x_nunique": math.nan,
                "y_nunique": math.nan,
                "rho": float(np.mean(rhos)) if len(rhos) == len(PANELS) else math.nan,
                "pvalue_two_sided": math.nan,
                "status": "estimable_posthoc_macro"
                if len(rhos) == len(PANELS)
                else "undefined_panel_component",
                "interpretation_constraint": "macro_effect_only_no_macro_pvalue_hypothesis_generating_only",
            }
        )
    return pd.DataFrame(rows)


def preregistered_estimability(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prereg_scores = (
        "epistemic_confidence",
        "aleatoric_confidence",
        "combined_confidence_official",
        "predicted_magnitude_rms",
    )
    for panel in PANELS:
        subset = joined.loc[joined["panel"] == panel]
        for score in prereg_scores:
            for endpoint, label, expected_direction in PRIMARY_ENDPOINTS:
                result = safe_spearman(subset[score], subset[endpoint])
                rows.append(
                    {
                        "analysis_role": "E158_preregistered_estimability_forensic_check",
                        "panel": panel,
                        "score": score,
                        "endpoint": endpoint,
                        "endpoint_label": label,
                        "expected_direction": expected_direction,
                        **result,
                        "rho_serialization_rule": "blank_when_undefined_never_zero_filled",
                    }
                )
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "未定义"
    return f"{float(value):.{digits}f}"


def build_svg(score_summary: pd.DataFrame, chain_summary: pd.DataFrame) -> str:
    p3_log = score_summary.query("panel == 'Norman_P3' and score == 'log_prob'").iloc[0]
    p4_log = score_summary.query("panel == 'Norman_P4' and score == 'log_prob'").iloc[0]
    evidence = chain_summary.iloc[0]["evidence_exp_float32_min"]
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="760" viewBox="0 0 1800 760">
  <rect x="0" y="0" width="1800" height="760" fill="#ffffff"/>
  <style>
    .title {{ font: 700 34px Arial, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; fill: #171717; }}
    .subtitle {{ font: 400 19px Arial, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; fill: #555555; }}
    .head {{ font: 700 22px Arial, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; fill: #171717; }}
    .body {{ font: 400 18px Arial, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; fill: #333333; }}
    .mono {{ font: 600 17px "DejaVu Sans Mono", Consolas, monospace; fill: #17365d; }}
    .arrow {{ stroke: #777777; stroke-width: 3; fill: none; marker-end: url(#arrow); }}
  </style>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#777777"/>
    </marker>
  </defs>
  <text x="80" y="70" class="title">E159 · PRESCRIBE 官方置信度饱和链</text>
  <text x="80" y="108" class="subtitle">事后取证；只使用 E157 锁定分数与 E158 已物化任务指标；未读取 raw H5AD / X</text>

  <rect x="70" y="170" width="260" height="190" rx="12" fill="#f7f9fb" stroke="#8da6bd" stroke-width="2"/>
  <text x="95" y="212" class="head">1  原始密度仍有差异</text>
  <text x="95" y="254" class="body">P3：24 个不同值</text>
  <text x="95" y="286" class="mono">{p3_log['min']:.3f} ～ {p3_log['max']:.3f}</text>
  <text x="95" y="320" class="body">P4：24 个不同值</text>
  <text x="95" y="350" class="mono">{p4_log['min']:.3f} ～ {p4_log['max']:.3f}</text>

  <path d="M330 265 L390 265" class="arrow"/>
  <rect x="400" y="170" width="230" height="190" rx="12" fill="#ffffff" stroke="#9c9c9c" stroke-width="2"/>
  <text x="425" y="212" class="head">2  exp 预算</text>
  <text x="425" y="262" class="mono">log_prob + 10</text>
  <text x="425" y="306" class="body">仍全部低于 −30</text>

  <path d="M630 265 L690 265" class="arrow"/>
  <rect x="700" y="170" width="230" height="190" rx="12" fill="#fff8f5" stroke="#c07a55" stroke-width="2"/>
  <text x="725" y="212" class="head">3  下界截断</text>
  <text x="725" y="262" class="mono">clamp → −30</text>
  <text x="725" y="306" class="body">48/48 任务相同</text>

  <path d="M930 265 L990 265" class="arrow"/>
  <rect x="1000" y="170" width="250" height="190" rx="12" fill="#ffffff" stroke="#9c9c9c" stroke-width="2"/>
  <text x="1025" y="212" class="head">4  极小证据</text>
  <text x="1025" y="262" class="mono">exp(−30)</text>
  <text x="1025" y="306" class="mono">{evidence:.4e}</text>

  <path d="M1250 265 L1310 265" class="arrow"/>
  <rect x="1320" y="170" width="410" height="190" rx="12" fill="#fff8f5" stroke="#c07a55" stroke-width="2"/>
  <text x="1345" y="212" class="head">5  float32 消减</text>
  <text x="1345" y="258" class="mono">1 − 1/(1+evidence) = 0</text>
  <text x="1345" y="306" class="body">posterior 精确退回 prior</text>
  <text x="1345" y="338" class="body">预测、熵、官方置信度全部相同</text>

  <path d="M900 390 L900 455" class="arrow"/>
  <rect x="330" y="470" width="1140" height="160" rx="14" fill="#f5f5f5" stroke="#555555" stroke-width="2"/>
  <text x="370" y="520" class="head">统计后果</text>
  <text x="370" y="562" class="body">P3 / P4 内官方分数均为常数，Spearman 与 10,000 次 bootstrap 无法产生有效重复。</text>
  <text x="370" y="602" class="body">结论应写“主 gate 不可估计且失败”，不能写 ρ=0，也不能用事后 raw log_prob 救回。</text>
  <text x="80" y="704" class="subtitle">白底证据图 · 数值来自 E159 固定输入；图中箭头表示实际源码计算顺序</text>
</svg>
'''


def markdown_table_for_scores(score_summary: pd.DataFrame) -> str:
    labels = {
        "log_prob": "raw log_prob",
        "epistemic_confidence": "epistemic",
        "aleatoric_confidence": "aleatoric",
        "combined_confidence_official": "official combined",
        "predicted_magnitude_rms": "predicted magnitude",
    }
    lines = [
        "| Panel | 分数 | min | max | 标准差(ddof=0) | 不同值数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in score_summary.itertuples(index=False):
        lines.append(
            f"| {row.panel} | {labels[row.score]} | {row.min:.12g} | "
            f"{row.max:.12g} | {row.std_population_ddof0:.12g} | {row.nunique} |"
        )
    return "\n".join(lines)


def markdown_table_for_posthoc(posthoc: pd.DataFrame) -> str:
    subset = posthoc.loc[
        posthoc["scope"].isin([*PANELS, "equal_panel_macro"])
    ]
    lines = [
        "| 终点 | P3 ρ | P4 ρ | 等权 macro ρ |",
        "|---|---:|---:|---:|",
    ]
    for endpoint in [
        *(name for name, _, _ in PRIMARY_ENDPOINTS),
        *(name for name, _, _ in RAW_SENSITIVITY_ENDPOINTS),
    ]:
        rows = subset.loc[subset["endpoint"] == endpoint].set_index("scope")
        label = rows.iloc[0]["endpoint_label"]
        lines.append(
            f"| {label} | {fmt(rows.loc['Norman_P3', 'rho'])} | "
            f"{fmt(rows.loc['Norman_P4', 'rho'])} | "
            f"{fmt(rows.loc['equal_panel_macro', 'rho'])} |"
        )
    return "\n".join(lines)


def build_report(
    score_summary: pd.DataFrame,
    pca_summary: pd.DataFrame,
    chain_summary: pd.DataFrame,
    posthoc: pd.DataFrame,
    e158_status: dict[str, Any],
    input_manifest: pd.DataFrame,
) -> str:
    all_pca_constant = bool(pca_summary["constant"].all())
    all_clamped = bool(np.allclose(chain_summary["clamp_fraction"], 1.0))
    input_hash_lines = "\n".join(
        f"- `{row.role}`：`{row.sha256}`" for row in input_manifest.itertuples(index=False)
    )
    return f"""# E159 PRESCRIBE 饱和取证报告

生成时间：{now_iso()}  
分析性质：**测试真值解封后的取证分析，不是预注册验证**

## 一、结论

E158 `attempt_001` 的主 gate **不可估计且已经失败**。失败不是 Spearman 或 bootstrap 程序把相关系数算错，而是 P3、P4 面板内所有预注册评分都没有任务间方差。未定义的相关系数在本报告中保留为空值并标记 `undefined_constant_score`，没有写成 0。

P3/P4 不能通过改用 raw `log_prob`、修改 clamp、增加微小抖动或改变终点来“救回”。这些操作发生在测试真值解封之后，只能形成下一项实验的新假设。

## 二、只读边界

本次程序只读取：

1. E157 两个面板已经锁定的 label-only 分数表；
2. E158 `attempt_001` 已经物化的任务指标、运行状态与解封事件；
3. 五个固定 PRESCRIBE 源码文件。

程序没有打开任何 `.h5ad`、HDF5、Zarr 或 Loom 文件，没有读取 raw H5AD，也没有物化任何表达矩阵 `X`。E158 状态原文显示：`phase={e158_status.get('phase')}`、`test_data_unsealed={e158_status.get('test_data_unsealed')}`、`test_X_rows_materialized={e158_status.get('test_X_rows_materialized')}`。这些是对 E158 已发生事件的读取，不是 E159 再次访问测试数据。

## 三、每个面板的分数动态范围

{markdown_table_for_scores(score_summary)}

两面板的 raw `log_prob` 各有24个不同值；epistemic、aleatoric、official combined 和 predicted magnitude 在各面板均只有1个值。十个预测 PCA 坐标是否全部为常数：**{all_pca_constant}**。

## 四、饱和计算链

E157 固定配置为 `certainty_budget=exp`、`output_dim=10`、`bound=30`。固定源码执行以下计算：

```text
raw log_prob
  → log_prob + 10
  → clamp(lower=-30, upper=30)
  → exp(log_evidence)
  → prior_proportion = 1 / (1 + evidence)
  → evidence_proportion = 1 - prior_proportion
  → epistemic = 10 × evidence_proportion + 10
```

两个面板的 `log_prob + 10` 仍全部小于 −30，截断比例是否为100%：**{all_clamped}**。截断后每个任务都是 −30，float32 的 `exp(-30)` 约为 `{chain_summary.iloc[0]['evidence_exp_float32_min']:.17g}`。源码写法 `1 - 1/(1+evidence)` 在 float32 中得到精确的 0，posterior 因而退回 prior。

采用数值上更稳定的 `evidence/(1+evidence)` 可以保留约 `9.36e-14` 的极小量，但它不能恢复已被 clamp 抹掉的任务顺序，也不足以把 E158 变成有效的前瞻验证。若修改公式，必须作为新模型版本另行冻结和验证。

![E159 饱和链](figures/E159_SATURATION_FLOW.svg)

## 五、raw log_prob 的事后探索

下表全部是 **post-hoc、hypothesis-generating only**：

{markdown_table_for_posthoc(posthoc)}

这些弱到中等的相关趋势只说明 raw density 值得在完全独立、尚未解封的数据上重新预注册。它们不能计入 E158 主结果，不能表述成 P3/P4 外部验证成功。raw sensitivity 的 Pearson 指标还呈相反方向，更不能选择性报告有利终点。

## 六、E158 的正式处理口径

- `attempt_001` 保持原样，不删除、不覆盖。
- 主结果写作：**官方 PRESCRIBE 分数在严格未见基因 OOD 区间完全退化，主关联统计不可估计，预注册 gate 失败。**
- 不把未定义写成 `ρ=0`。
- 不运行内容相同的 `attempt_002`。
- P3/P4 只允许作为新分数设计的开发证据；下一次确认性检验必须使用新的未解封数据和预先冻结的非退化门槛。

## 七、可复核文件

- `tables/E159_SCORE_SUMMARY.csv`：每个面板、每个分数的 min/max/std/nunique。
- `tables/E159_PCA_PREDICTION_SUMMARY.csv`：十个预测 PCA 坐标的退化检查。
- `tables/E159_SCALER_CHAIN_TASKS.csv`：逐任务 float32 计算链。
- `tables/E159_SCALER_CHAIN_SUMMARY.csv`：逐面板截断摘要。
- `tables/E159_PREREGISTERED_ESTIMABILITY.csv`：未定义相关显式留空。
- `tables/E159_POSTHOC_SPEARMAN.csv`：事后 Spearman，含 P3、P4、pooled 与等权 macro。
- `tables/E159_POSTHOC_JOINED_TASKS.csv`：锁定分数与已物化任务指标的一对一连接。
- `tables/E159_JOIN_AUDIT.csv`：E157 与 E158 携带分数的一致性检查。
- `INPUT_MANIFEST.csv`、`OUTPUT_MANIFEST.csv`：输入、输出 SHA-256。
- `RUN_STATUS.json`：边界、状态和关键判定。

## 八、输入 SHA-256

{input_hash_lines}
"""


def output_manifest(result_dir: Path, paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        rows.append(
            {
                "relative_path": str(path.relative_to(result_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo_default = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo_default)
    parser.add_argument(
        "--prescribe-root",
        type=Path,
        default=Path("/home/yyf/archive/external/PRESCRIBE"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_default
        / "docs/实验结果/E159_prescribe_saturation_forensics_20260714",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    prescribe = args.prescribe_root.resolve()
    result_dir = args.output_dir.resolve()
    tables_dir = result_dir / "tables"
    figures_dir = result_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    status_path = result_dir / "RUN_STATUS.json"
    started_at = now_iso()
    runner_sha = sha256_file(Path(__file__).resolve())
    base_status: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "phase": "running_post_unseal_forensics",
        "started_at": started_at,
        "analysis_timing": ANALYSIS_TIMING,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": runner_sha,
        "certainty_budget": CERTAINTY_BUDGET,
        "output_dim": OUTPUT_DIM,
        "log_scale": LOG_SCALE,
        "evidence_bound": EVIDENCE_BOUND,
        "read_boundary": {
            "allowlisted_inputs_only": True,
            "raw_h5ad_accessed": False,
            "hdf5_container_opened": False,
            "test_X_materialized_by_E159": False,
            "expression_matrix_read_by_E159": False,
        },
    }
    write_json(status_path, base_status)

    try:
        specs = build_input_specs(repo, prescribe)
        payloads, input_manifest = load_allowlisted_inputs(specs)
        assert_fixed_source_contract(payloads)

        p3 = csv_from_payload(payloads["E157_locked_scores_P3"])
        p4 = csv_from_payload(payloads["E157_locked_scores_P4"])
        metrics = csv_from_payload(payloads["E158_materialized_task_metrics"])
        e158_status = json_from_payload(payloads["E158_attempt_status"])
        e158_unseal = json_from_payload(payloads["E158_unseal_event"])
        if e158_status.get("test_data_unsealed") is not True:
            raise RuntimeError("E159 is only valid as post-unseal forensics")
        if "failed_after_irreversible_test_unseal" not in str(e158_status.get("phase")):
            raise RuntimeError("Unexpected E158 attempt phase")

        joined, join_audit = validate_and_join(p3, p4, metrics)
        locked = pd.concat([p3, p4], ignore_index=True)
        score_summary = summarize_scores(locked)
        pca_summary = summarize_pca(locked)
        chain_tasks, chain_summary = reconstruct_scaler_chain(locked)
        posthoc = posthoc_spearman(joined)
        estimability = preregistered_estimability(joined)

        if not score_summary.loc[
            score_summary["score"] != "log_prob", "constant"
        ].all():
            raise RuntimeError("Expected official E157 scores to be constant")
        if not np.allclose(chain_summary["clamp_fraction"], 1.0):
            raise RuntimeError("Expected all tasks to hit the lower evidence clamp")
        if not estimability["rho"].isna().all():
            raise RuntimeError("Undefined preregistered associations were unexpectedly estimable")

        input_manifest_path = result_dir / "INPUT_MANIFEST.csv"
        score_summary_path = tables_dir / "E159_SCORE_SUMMARY.csv"
        pca_summary_path = tables_dir / "E159_PCA_PREDICTION_SUMMARY.csv"
        chain_tasks_path = tables_dir / "E159_SCALER_CHAIN_TASKS.csv"
        chain_summary_path = tables_dir / "E159_SCALER_CHAIN_SUMMARY.csv"
        posthoc_path = tables_dir / "E159_POSTHOC_SPEARMAN.csv"
        joined_path = tables_dir / "E159_POSTHOC_JOINED_TASKS.csv"
        estimability_path = tables_dir / "E159_PREREGISTERED_ESTIMABILITY.csv"
        join_audit_path = tables_dir / "E159_JOIN_AUDIT.csv"
        svg_path = figures_dir / "E159_SATURATION_FLOW.svg"
        report_path = result_dir / "E159_事后饱和取证报告.md"

        write_csv(input_manifest, input_manifest_path)
        write_csv(score_summary, score_summary_path)
        write_csv(pca_summary, pca_summary_path)
        write_csv(chain_tasks, chain_tasks_path)
        write_csv(chain_summary, chain_summary_path)
        write_csv(posthoc, posthoc_path)
        write_csv(joined, joined_path)
        write_csv(estimability, estimability_path)
        write_csv(join_audit, join_audit_path)
        svg_path.write_text(build_svg(score_summary, chain_summary), encoding="utf-8")
        report_path.write_text(
            build_report(
                score_summary,
                pca_summary,
                chain_summary,
                posthoc,
                e158_status,
                input_manifest,
            ),
            encoding="utf-8",
        )

        generated = [
            input_manifest_path,
            score_summary_path,
            pca_summary_path,
            chain_tasks_path,
            chain_summary_path,
            posthoc_path,
            joined_path,
            estimability_path,
            join_audit_path,
            svg_path,
            report_path,
        ]
        out_manifest = output_manifest(result_dir, generated)
        out_manifest_path = result_dir / "OUTPUT_MANIFEST.csv"
        write_csv(out_manifest, out_manifest_path)

        official = score_summary[score_summary["score"] != "log_prob"]
        raw_posthoc = posthoc[
            (posthoc["scope"].isin(PANELS))
            & (posthoc["endpoint"].isin(name for name, _, _ in PRIMARY_ENDPOINTS))
        ]
        completed = {
            **base_status,
            "phase": "complete_post_unseal_saturation_forensics",
            "completed_at": now_iso(),
            "input_manifest_sha256": sha256_file(input_manifest_path),
            "output_manifest_sha256": sha256_file(out_manifest_path),
            "output_manifest_excludes": [
                "OUTPUT_MANIFEST.csv (self-hash recursion)",
                "RUN_STATUS.json (written after manifest)",
            ],
            "e158_source_attempt": {
                "phase": e158_status.get("phase"),
                "test_data_unsealed": e158_status.get("test_data_unsealed"),
                "test_X_rows_materialized": e158_status.get("test_X_rows_materialized"),
                "error": e158_status.get("error"),
                "unseal_event_sha256_reported_by_E158": e158_status.get(
                    "unseal_event_sha256"
                ),
                "unseal_event_top_level_keys": sorted(e158_unseal.keys()),
            },
            "forensic_findings": {
                "n_panels": len(PANELS),
                "n_tasks": len(joined),
                "all_official_and_magnitude_scores_constant_within_panel": bool(
                    official["constant"].all()
                ),
                "all_ten_predicted_pca_coordinates_constant_within_panel": bool(
                    pca_summary["constant"].all()
                ),
                "all_tasks_hit_lower_clamp": bool(
                    np.allclose(chain_summary["clamp_fraction"], 1.0)
                ),
                "unstable_float32_evidence_proportion_exactly_zero": bool(
                    (chain_tasks[
                        "evidence_proportion_unstable_1_minus_prior_float32"
                    ]
                    == 0.0).all()
                ),
                "E158_primary_gate_estimable": False,
                "E158_primary_gate_result": "failed_non_estimable_constant_score",
                "undefined_rho_zero_filled": False,
                "P3_P4_can_be_rescued_posthoc": False,
                "raw_log_prob_analysis_role": "posthoc_hypothesis_generating_only",
            },
            "posthoc_primary_panel_rhos": raw_posthoc[
                ["scope", "endpoint", "rho", "status"]
            ].to_dict(orient="records"),
            "read_boundary": {
                **base_status["read_boundary"],
                "allowlisted_input_count": len(specs),
                "allowed_suffixes_observed": sorted(
                    {spec.path.suffix.lower() for spec in specs}
                ),
            },
            "outputs": [
                *out_manifest["relative_path"].tolist(),
                "OUTPUT_MANIFEST.csv",
                "RUN_STATUS.json",
            ],
        }
        write_json(status_path, completed)
        print(json.dumps(completed, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failed = {
            **base_status,
            "phase": "failed_post_unseal_forensics_preserve_attempt",
            "failed_at": now_iso(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(status_path, failed)
        raise


if __name__ == "__main__":
    sys.exit(main())

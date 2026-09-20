from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/scripts/summarize_e205_formal_results.py"
SPEC = importlib.util.spec_from_file_location("e205_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class E205ResultSummaryTests(unittest.TestCase):
    def test_summary_renders_supported_and_negative_gates_without_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            (evaluation / "E205_FORMAL_EVALUATION_STATUS.json").write_text(
                json.dumps(
                    {
                        "execution_status": "PASS",
                        "primary_increment_status": "NOT_SUPPORTED",
                        "registered_family_certificate_status": "SUPPORTED",
                        "n_primary_tasks": 1808,
                        "registered_family_lower_bound_violations": 0,
                        "registered_family_identity_failures": 0,
                        "registered_family_lower_tightness_median": 0.25,
                        "certificate_priority_router_status": "NOT_SUPPORTED",
                        "architecture_aware_router_status": "NOT_SUPPORTED",
                        "context_holdout_router_status": "NOT_SUPPORTED",
                    }
                ),
                encoding="utf-8",
            )
            predictors = list(MODULE.PREDICTOR_LABELS)
            scopes = ["pooled", "K562", "RPE1", "hepg2", "jurkat"]
            pd.DataFrame(
                [
                    {
                        "scope": scope,
                        "predictor": name,
                        "spearman": 0.1 + index / 100 + scope_index / 200,
                    }
                    for scope_index, scope in enumerate(scopes)
                    for index, name in enumerate(predictors)
                ]
            ).to_csv(evaluation / "E205_RISK_ASSOCIATIONS.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "scope": scope,
                        "predictor": name,
                        "budget": budget,
                        "oracle_normalized_utility": (
                            0.2 + index / 100 + scope_index / 200 + budget / 20
                        ),
                    }
                    for scope_index, scope in enumerate(scopes)
                    for index, name in enumerate(predictors)
                    for budget in (0.05, 0.10, 0.20, 0.30)
                ]
            ).to_csv(evaluation / "E205_REVIEW_UTILITY.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "measure": "delta_spearman",
                        "estimate": -0.02,
                        "ci95_lower": -0.05,
                        "ci95_upper": 0.01,
                    },
                    {
                        "measure": "delta_utility_20",
                        "estimate": -0.01,
                        "ci95_lower": -0.03,
                        "ci95_upper": 0.02,
                    },
                ]
            ).to_csv(evaluation / "E205_INCREMENTAL_INTERVALS.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "unit": "task",
                        "scope": scope,
                        "quantile": number / 10,
                        "certified_high_recall": number / 20 + scope_index / 100,
                        "certified_high_coverage": number / 30 + scope_index / 100,
                    }
                    for scope_index, scope in enumerate(scopes)
                    for number in range(1, 10)
                ]
            ).to_csv(
                evaluation / "E205_REGISTERED_CERTIFICATE_CURVES.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "scope": scope,
                        "predictor": predictor,
                        "budget": budget,
                        "oracle_normalized_utility": (
                            0.25 + index / 100 + scope_index / 200 + budget / 20
                        ),
                    }
                    for scope_index, scope in enumerate(scopes)
                    for index, predictor in enumerate(MODULE.REGISTERED_ROUTER_LABELS)
                    for budget in (0.05, 0.10, 0.20, 0.30)
                ]
            ).to_csv(
                evaluation / "E205_REGISTERED_ROUTING_UTILITY.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "predictor": predictor,
                        "measure": measure,
                        "estimate": 0.01 if measure == "delta_spearman" else -0.01,
                        "ci95_lower": -0.01 if measure == "delta_spearman" else -0.03,
                        "ci95_upper": 0.03 if measure == "delta_spearman" else 0.02,
                    }
                    for predictor in (
                        "context_holdout_router",
                        "architecture_aware_router",
                        "certificate_priority_q80",
                    )
                    for measure in ("delta_spearman", "delta_utility_20")
                ]
            ).to_csv(
                evaluation / "E205_REGISTERED_ROUTING_INTERVALS.csv", index=False
            )
            report = evaluation / "REPORT.md"
            figure = evaluation / "overview.svg"
            with patch(
                "sys.argv",
                [
                    str(SCRIPT),
                    "--evaluation-dir",
                    str(evaluation),
                    "--output-report",
                    str(report),
                    "--output-figure",
                    str(figure),
                ],
            ):
                MODULE.main()

            text = report.read_text(encoding="utf-8")
            self.assertIn("注册家族证书 | SUPPORTED", text)
            self.assertIn("排序增量 | NOT_SUPPORTED", text)
            self.assertIn("排序模块不得作为主要胜利", text)
            self.assertIn("证书优先路由未确认复核增量", text)
            self.assertTrue(figure.is_file() and figure.stat().st_size > 0)
            self.assertTrue(figure.with_suffix(".pdf").is_file())
            self.assertTrue(figure.with_suffix(".png").is_file())
            self.assertTrue(
                (
                    evaluation
                    / "figures/E205_CONTEXT_RESOLVED_RESULTS.svg"
                ).is_file()
            )
            self.assertTrue(
                (
                    evaluation
                    / "figures/E205_OPERATING_CHARACTERISTICS.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    evaluation
                    / "figures/E205_CERTIFICATE_PRIORITY_ROUTER.png"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()

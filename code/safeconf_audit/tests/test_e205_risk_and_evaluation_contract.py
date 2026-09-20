from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def load_script(module_name: str, filename: str):
    path = ROOT / "tools/scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RISK = load_script("e205_pretruth_risk", "run_e205_pretruth_risk_features.py")
EVALUATION = load_script("e205_formal_evaluation", "run_e205_formal_evaluation.py")


class E205PretruthRiskTests(unittest.TestCase):
    def test_rank_fusion_is_the_frozen_four_to_one_rule(self) -> None:
        magnitude = np.asarray([0.0, 10.0, 20.0])
        safeconf = np.asarray([20.0, 10.0, 0.0])

        observed = RISK.rank_fusion_4_to_1(magnitude, safeconf)

        np.testing.assert_allclose(observed, np.asarray([7.0, 10.0, 13.0]) / 15.0)

    def test_rank_fusion_rejects_misaligned_or_nonfinite_inputs(self) -> None:
        with self.assertRaises(RISK.RiskFailure):
            RISK.rank_fusion_4_to_1(np.asarray([1.0, 2.0]), np.asarray([1.0]))
        with self.assertRaises(RISK.RiskFailure):
            RISK.rank_fusion_4_to_1(
                np.asarray([1.0, np.nan]), np.asarray([1.0, 2.0])
            )

    def test_architecture_router_is_upward_only_and_uses_frozen_coefficients(self) -> None:
        magnitude = np.asarray([0.0, 1.0, 2.0, 3.0])
        safeconf = np.asarray([3.0, 2.0, 1.0, 0.0])
        disagreement = np.asarray([1.0, 3.0, 0.0, 2.0])

        observed = RISK.architecture_aware_router(
            magnitude, safeconf, disagreement
        )
        m = np.asarray([0.25, 0.50, 0.75, 1.00])
        s = np.asarray([1.00, 0.75, 0.50, 0.25])
        d = np.asarray([0.50, 1.00, 0.25, 0.75])
        expected = m + 0.50 * np.maximum(s - m, 0.0) + 0.125 * np.maximum(
            d - m, 0.0
        )

        np.testing.assert_allclose(observed, expected)
        self.assertTrue(np.all(observed >= m))

    def test_architecture_router_rejects_misaligned_inputs(self) -> None:
        with self.assertRaises(RISK.RiskFailure):
            RISK.architecture_aware_router(
                np.asarray([1.0, 2.0]),
                np.asarray([1.0]),
                np.asarray([1.0, 2.0]),
            )

    def test_context_holdout_router_uses_conservative_equal_corrections(self) -> None:
        magnitude = np.asarray([0.0, 1.0, 2.0, 3.0])
        safeconf = np.asarray([3.0, 2.0, 1.0, 0.0])
        disagreement = np.asarray([1.0, 3.0, 0.0, 2.0])

        observed = RISK.context_holdout_router(
            magnitude, safeconf, disagreement
        )
        m = np.asarray([0.25, 0.50, 0.75, 1.00])
        s = np.asarray([1.00, 0.75, 0.50, 0.25])
        d = np.asarray([0.50, 1.00, 0.25, 0.75])
        expected = m + 0.125 * np.maximum(s - m, 0.0) + 0.125 * np.maximum(
            d - m, 0.0
        )
        np.testing.assert_allclose(observed, expected)

    def test_historical_nonnegative_router_uses_e218_frozen_weights(self) -> None:
        magnitude = np.asarray([0.0, 1.0, 2.0, 3.0])
        safeconf = np.asarray([3.0, 2.0, 1.0, 0.0])
        disagreement = np.asarray([1.0, 3.0, 0.0, 2.0])
        observed = RISK.historical_nonnegative_router(
            magnitude, safeconf, disagreement
        )
        m = np.asarray([0.25, 0.50, 0.75, 1.00])
        s = np.asarray([1.00, 0.75, 0.50, 0.25])
        d = np.asarray([0.50, 1.00, 0.25, 0.75])
        expected = (
            m
            + 0.5533545399558647 * np.maximum(s - m, 0.0)
            + 0.10610102453715037 * np.maximum(d - m, 0.0)
        )
        np.testing.assert_allclose(observed, expected)

    def test_certificate_priority_is_lexicographic_and_untuned(self) -> None:
        magnitude = np.asarray([0.1, 0.9, 0.2, 0.8])
        lower_bound = np.asarray([0.6, 0.4, 0.7, 0.3])

        observed = RISK.certificate_priority_score(
            magnitude, lower_bound, tau=0.5
        )

        certified = lower_bound > 0.5
        self.assertGreater(float(observed[certified].min()), float(observed[~certified].max()))
        self.assertGreater(float(observed[2]), float(observed[0]))
        self.assertGreater(float(observed[1]), float(observed[3]))

    def test_certificate_priority_rejects_nonfinite_inputs(self) -> None:
        with self.assertRaises(RISK.RiskFailure):
            RISK.certificate_priority_score(
                np.asarray([0.1, np.nan]), np.asarray([0.2, 0.3]), 0.25
            )

    def test_prediction_tree_fails_closed_on_truth_named_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "K562/seed_1").mkdir(parents=True)
            (root / "K562/seed_1/predictions.npy").write_bytes(b"prediction")
            RISK.ensure_no_truth_artifacts(root)

            forbidden = root / "K562/shared/target_truth.npy"
            forbidden.parent.mkdir(parents=True)
            forbidden.write_bytes(b"must not be present")
            with self.assertRaisesRegex(RISK.RiskFailure, "truth artifact"):
                RISK.ensure_no_truth_artifacts(root)

    def test_truth_named_directory_is_also_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "truth_cache").mkdir()
            with self.assertRaisesRegex(RISK.RiskFailure, "truth artifact"):
                RISK.ensure_no_truth_artifacts(root)


class E205EvaluationAuthorizationTests(unittest.TestCase):
    def authorization(self) -> dict:
        return {
            "experiment": "E205_cross_family_exphormer",
            "stage": "TARGET_TRUTH_REUSE_AUTHORIZATION",
            "status": "AUTHORIZED",
            "allow_released_e201_target_centroids": True,
            "risk_sealed_and_pushed_before_authorization": True,
            "pretruth_risk_status_sha256": "risk-status",
            "pretruth_risk_table_sha256": "risk-table",
            "e201_core_final_status_sha256": "e201-final",
            "risk_seal_commit": "1234567",
        }

    def test_authorization_accepts_only_the_exact_pretruth_artifacts(self) -> None:
        commit = EVALUATION.validate_authorization(
            self.authorization(), "risk-status", "risk-table", "e201-final"
        )
        self.assertEqual(commit, "1234567")

    def test_authorization_rejects_wrong_status_or_hash(self) -> None:
        wrong_status = self.authorization()
        wrong_status["status"] = "NOT_AUTHORIZED"
        with self.assertRaises(EVALUATION.EvaluationFailure):
            EVALUATION.validate_authorization(
                wrong_status, "risk-status", "risk-table", "e201-final"
            )

        wrong_hash = self.authorization()
        wrong_hash["pretruth_risk_table_sha256"] = "another-table"
        with self.assertRaises(EVALUATION.EvaluationFailure):
            EVALUATION.validate_authorization(
                wrong_hash, "risk-status", "risk-table", "e201-final"
            )

    def test_authorization_record_round_trips_without_opening_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authorization.json"
            path.write_text(json.dumps(self.authorization()), encoding="utf-8")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                EVALUATION.validate_authorization(
                    record, "risk-status", "risk-table", "e201-final"
                ),
                "1234567",
            )


class E205EvaluationMetricTests(unittest.TestCase):
    def test_family_error_identity_and_lower_bound(self) -> None:
        seed = np.asarray(
            [
                [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]],
                [[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]],
                [[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]],
                [[1.5, 1.5], [4.5, 4.5], [7.5, 7.5]],
            ],
            dtype=np.float32,
        )
        family = seed.mean(axis=0)
        gat_seed = seed + np.asarray([0.2, -0.1, 0.3, -0.2])[:, None, None]
        registered_members = np.concatenate([gat_seed, seed], axis=0)
        registered_family = registered_members.mean(axis=0)
        truth = np.asarray([[0.5, 1.0], [5.0, 3.0], [5.0, 10.0]], dtype=np.float32)
        disagreement = np.sqrt(
            np.mean(np.square(seed.astype(float) - family[None, :]), axis=(0, 2))
        )
        features = pd.DataFrame(
            {
                "task_id": ["a", "b", "c"],
                "family_disagreement": disagreement,
                "registered_family_disagreement": np.sqrt(
                    np.mean(
                        np.square(
                            registered_members.astype(float)
                            - registered_family.astype(float)[None, :]
                        ),
                        axis=(0, 2),
                    )
                ),
            }
        )
        result = EVALUATION.task_errors(
            features,
            {
                "E205_SEED_CENTROIDS.npy": seed,
                "E205_FAMILY_CENTROIDS.npy": family,
                "E201_GAT_SEED_CENTROIDS.npy": gat_seed,
                "E205_REGISTERED_FAMILY_CENTROIDS.npy": registered_family,
            },
            truth,
        )

        self.assertLess(float(result.family_identity_residual.abs().max()), 1e-12)
        self.assertTrue(
            np.all(
                result.family_disagreement.to_numpy(float)
                <= result.family_rms_error.to_numpy(float) + 1e-12
            )
        )
        self.assertLess(
            float(result.registered_family_identity_residual.abs().max()),
            1e-12,
        )
        self.assertTrue(
            np.all(
                result.registered_family_identity_tolerance.to_numpy(float)
                >= 1e-9
            )
        )
        self.assertTrue(
            np.all(
                result.registered_family_disagreement.to_numpy(float)
                <= result.registered_family_rms_error.to_numpy(float) + 1e-12
            )
        )

    def test_registered_certificate_curve_has_no_false_certificate(self) -> None:
        frame = pd.DataFrame(
            {
                "target": ["K562", "RPE1", "hepg2", "jurkat", "K562"],
                "condition": ["A", "A", "B", "C", "D"],
                "registered_family_rms_error": [0.2, 0.3, 0.4, 0.1, 0.5],
                "registered_family_disagreement": [0.1, 0.2, 0.35, 0.05, 0.45],
            }
        )
        curves = EVALUATION.certificate_curves(
            frame,
            [{"quantile": number / 10, "tau": 0.25} for number in range(1, 10)],
        )
        self.assertEqual(int(curves.n_false_certificates.sum()), 0)
        pooled = curves.loc[
            curves.unit.eq("task") & curves.scope.eq("pooled")
        ].iloc[0]
        self.assertEqual(int(pooled.n_true_high), 3)
        self.assertEqual(int(pooled.n_certified_high), 2)
        self.assertAlmostEqual(float(pooled.certified_high_recall), 2 / 3)

    def test_perfect_ranking_has_full_review_utility_and_capture(self) -> None:
        outcome = np.arange(1.0, 11.0)
        result = EVALUATION.review_metrics(
            outcome,
            outcome,
            np.asarray([f"task-{index}" for index in range(10)]),
            0.20,
        )

        self.assertEqual(result["n_selected"], 2)
        self.assertAlmostEqual(result["high_error_capture"], 1.0)
        self.assertAlmostEqual(result["oracle_normalized_utility"], 1.0)

    def test_bootstrap_rejects_informal_draw_count(self) -> None:
        with self.assertRaisesRegex(EVALUATION.EvaluationFailure, "at least 100"):
            EVALUATION.bootstrap_increment(pd.DataFrame(), 99)
        with self.assertRaisesRegex(EVALUATION.EvaluationFailure, "at least 100"):
            EVALUATION.bootstrap_registered_router(pd.DataFrame(), 99)


if __name__ == "__main__":
    unittest.main()

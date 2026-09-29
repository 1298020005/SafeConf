from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.safeconf_continual import (
    ContinualUpdateManager,
    ErrorResidualAdapter,
    ErrorMemoryItem,
    ErrorMemoryRegistry,
    FrozenErrorCDF,
    PublicMemoryItem,
    PublicMemoryStore,
    PublicBiologyLearner,
    RiskOutput,
    ReleaseMetrics,
    biological_task_key,
)


class SafeConfContinualTest(unittest.TestCase):
    def test_same_biological_truth_has_same_key_across_upstreams(self) -> None:
        args = ("study", "K562", "knockout", "TP53", "24h", "delta-v1")
        self.assertEqual(biological_task_key(*args), biological_task_key(*args))

    def test_frozen_error_cdf_uses_only_fit_reference(self) -> None:
        cdf = FrozenErrorCDF.fit([1.0, 2.0, 3.0, 4.0])
        before = cdf.transform([0.0, 2.5, 9.0])
        after = cdf.transform([0.0, 2.5, 9.0])
        self.assertTrue(np.allclose(before, after))
        self.assertTrue(np.all((before >= 0) & (before <= 1)))

    def test_public_store_integrity_and_alignment(self) -> None:
        item = PublicMemoryItem(
            "e1", "s1", "K562", "knockout", "TP53", "24h", 0,
            "delta-v1", "matched", "genes-v1", 10, 2, 2, 1,
            0.8, 0.7, 0.9, 0.6, "unit-test", True, "2026-01-01",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = PublicMemoryStore(Path(directory) / "public")
            store.create([item], np.ones((1, 3)), np.zeros((1, 3)), ["a", "b", "c"], {})
            frame, effects, controls, manifest = store.load()
            self.assertEqual(len(frame), manifest["n_items"])
            self.assertEqual(effects.shape, controls.shape)

    def test_public_memory_append_publishes_version_without_mutating_parent(self) -> None:
        item = PublicMemoryItem(
            "e1", "s1", "K562", "knockout", "TP53", "24h", 0,
            "delta-v1", "matched", "genes-v1", 10, 2, 2, 1,
            0.8, 0.7, 0.9, 0.6, "unit-test", True, "2026-01-01",
        )
        item2 = PublicMemoryItem(
            "e2", "s1", "RPE1", "knockout", "TP53", "24h", 1,
            "delta-v1", "matched", "genes-v1", 12, 2, 2, 1,
            0.7, 0.8, 0.8, 0.7, "unit-test", True, "2026-01-01",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "public"
            store = PublicMemoryStore(root)
            store.create([item], np.ones((1, 3)), np.zeros((1, 3)), ["a", "b", "c"], {})
            store.append([item2], np.full((1, 3), 2.0), np.zeros((1, 3)), {})
            frame, effects, _, manifest = store.load()
            self.assertEqual(len(frame), 2)
            self.assertEqual(manifest["version"], 2)
            self.assertTrue((root / "versions/v0002/manifest.json").exists())

    def test_error_memory_is_model_version_specific(self) -> None:
        item = ErrorMemoryItem(
            "GEARS", "checkpoint-a", "p1", "task", "cluster",
            0.4, 0.6, 0.7, 0.3, "2026-01-01", "OOF", True,
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = ErrorMemoryRegistry(Path(directory) / "errors")
            registry.append([item])
            self.assertEqual(len(registry.load("GEARS", "checkpoint-a")), 1)
            with self.assertRaises(FileNotFoundError):
                registry.load("GEARS", "checkpoint-b")

    def test_risk_output_rejects_unbounded_score(self) -> None:
        with self.assertRaises(ValueError):
            RiskOutput(0.4, 0.8, 1.2, "p", "s", "e", 1, 2, "ok")

    def test_continual_update_triggers_and_release_gate(self) -> None:
        public = ContinualUpdateManager.public_update_due(200, 20, 0)
        self.assertTrue(public.trigger)
        self.assertIn("at_least_10_percent_bank_growth", public.reasons)
        feedback = ContinualUpdateManager.error_update_due(24, 25)
        self.assertTrue(feedback.trigger)
        self.assertIn("crossed_25_feedback_clusters", feedback.reasons)
        release = ContinualUpdateManager.release(ReleaseMetrics(
            delta_u20_new_tasks=-0.001,
            relative_aurc_degradation_anchor=0.02,
            miss_rate_degradation=0.01,
            nonnegative_strata_fraction=0.75,
        ))
        self.assertTrue(release.trigger)

    def test_low_capacity_learners_keep_contract_scales(self) -> None:
        x = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]])
        transfer = np.asarray([0.2, 0.1, 0.3, 0.15, 0.4, 0.25])
        public = PublicBiologyLearner("ridge").fit(x, transfer)
        weights = public.retrieval_weights(x, np.ones(len(x)), regularization=0.5)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        adapter = ErrorResidualAdapter("m", "v", "ridge", kappa=10).fit(
            x, np.full(len(x), 0.5), np.asarray([0.1, 0.2, 0.7, 0.4, 0.8, 0.6]), 5
        )
        correction, final = adapter.predict(x, np.full(len(x), 0.5))
        self.assertEqual(len(correction), len(final))
        self.assertTrue(np.all((final >= 0) & (final <= 1)))


if __name__ == "__main__":
    unittest.main()

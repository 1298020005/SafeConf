"""Release-gate semantics, including unavailable sparse-feedback metrics."""
from dataclasses import replace
import unittest

from tools.safeconf_continual.update import ContinualUpdateManager, ReleaseMetrics


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.valid = ReleaseMetrics(0.01, 0.0, 0.0, 0.8)

    def test_valid_release_and_registered_boundaries_pass(self):
        for metrics in (self.valid, ReleaseMetrics(-0.005, 0.05, 0.02, 0.60),
                        replace(self.valid, nonnegative_strata_fraction=1.0)):
            with self.subTest(metrics=metrics):
                decision = ContinualUpdateManager.release(metrics)
                self.assertTrue(decision.trigger)
                self.assertEqual(decision.reasons, ("release_gate_passed",))

    def test_original_finite_failure_conditions_remain(self):
        failures = (
            ("delta_u20_new_tasks", -0.0051, "new_task_u20_noninferiority_failed"),
            ("relative_aurc_degradation_anchor", 0.0501, "anchor_aurc_failed"),
            ("miss_rate_degradation", 0.0201, "high_risk_miss_rate_failed"),
            ("nonnegative_strata_fraction", 0.5999, "strata_consistency_failed"),
        )
        for field, value, reason in failures:
            with self.subTest(field=field):
                decision = ContinualUpdateManager.release(replace(self.valid, **{field: value}))
                self.assertFalse(decision.trigger)
                self.assertEqual(decision.reasons, (reason,))

    def test_each_nonfinite_metric_rejects_an_otherwise_valid_release(self):
        for field in self.valid.__dataclass_fields__:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    decision = ContinualUpdateManager.release(replace(self.valid, **{field: value}))
                    self.assertFalse(decision.trigger)
                    self.assertEqual(decision.reasons, (f"nonfinite_{field}",))

    def test_all_unavailable_metrics_report_all_rejection_reasons(self):
        decision = ContinualUpdateManager.release(ReleaseMetrics(*([float("nan")] * 4)))
        self.assertFalse(decision.trigger)
        self.assertEqual(decision.reasons, tuple(
            f"nonfinite_{field}" for field in self.valid.__dataclass_fields__))

    def test_fraction_outside_unit_interval_is_invalid(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value):
                decision = ContinualUpdateManager.release(
                    replace(self.valid, nonnegative_strata_fraction=value))
                self.assertFalse(decision.trigger)
                self.assertIn("strata_fraction_out_of_range", decision.reasons)

    def test_existing_update_triggers_remain(self):
        self.assertFalse(ContinualUpdateManager.public_update_due(1000, 24, 0).trigger)
        self.assertEqual(ContinualUpdateManager.public_update_due(1000, 25, 0).reasons,
                         ("at_least_25_new_clusters",))
        self.assertEqual(ContinualUpdateManager.public_update_due(100, 10, 0).reasons,
                         ("at_least_10_percent_bank_growth",))
        self.assertEqual(ContinualUpdateManager.public_update_due(1000, 0, 1).reasons,
                         ("new_independent_study",))
        self.assertFalse(ContinualUpdateManager.error_update_due(10, 24).trigger)
        self.assertEqual(ContinualUpdateManager.error_update_due(24, 51).reasons,
                         ("crossed_25_feedback_clusters", "crossed_50_feedback_clusters"))
        self.assertEqual(ContinualUpdateManager.error_update_due(149, 150).reasons,
                         ("additional_50_feedback_clusters",))


if __name__ == "__main__":
    unittest.main()

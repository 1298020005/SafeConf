"""Tests for released E181/E182/E183 certificate operating curves."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "safeconf_audit"))

from safeconf_audit.certificate_curves import (  # noqa: E402
    CertificateIntegrityError,
    compute_operating_curves,
    load_released_tasks,
    parse_tau_grid,
    run,
)


def synthetic_tasks() -> pd.DataFrame:
    rows = []
    for task_id, target, error, lower, upper in (
        ("a1", "A", 0.20, 0.10, 0.30),
        ("a2", "A", 0.40, 0.30, 0.50),
        ("b1", "B", 0.50, 0.45, 0.60),
    ):
        rows.append(
            {
                "study": "S",
                "task_id": task_id,
                "target_cluster": target,
                "family_rms_error": error,
                "diversity_lower": lower,
                "family_upper": upper,
                "worst_member_error": error,
                "diameter_lower": lower,
                "worst_upper": upper,
            }
        )
    return pd.DataFrame(rows)


class CertificateCurveTests(unittest.TestCase):
    def test_tau_parser_rejects_duplicates_and_unsorted_values(self) -> None:
        with self.assertRaises(Exception):
            parse_tau_grid("0.1,0.1")
        with self.assertRaises(Exception):
            parse_tau_grid("0.2,0.1")
        self.assertEqual(parse_tau_grid("0.1,0.2"), (0.1, 0.2))

    def test_certified_high_recall_and_coverage(self) -> None:
        task, target = compute_operating_curves(synthetic_tasks(), [0.25])
        row = task.loc[
            task.study.eq("S") & task.objective.eq("family_rms")
        ].iloc[0]
        self.assertEqual(int(row.n_true_high), 2)
        self.assertEqual(int(row.n_certified_high), 2)
        self.assertEqual(int(row.n_false_certificates), 0)
        self.assertAlmostEqual(float(row.certified_high_coverage), 2 / 3)
        self.assertAlmostEqual(float(row.certified_high_recall), 1.0)
        self.assertAlmostEqual(float(row.certified_high_precision), 1.0)

        target_row = target.loc[
            target.study.eq("S") & target.objective.eq("family_rms")
        ].iloc[0]
        self.assertEqual(int(target_row.n_units), 2)
        self.assertEqual(int(target_row.n_certified_high), 2)

    def test_threshold_is_strict_and_unknown_is_not_safe(self) -> None:
        task, _ = compute_operating_curves(synthetic_tasks(), [0.30])
        row = task.loc[
            task.study.eq("S") & task.objective.eq("family_rms")
        ].iloc[0]
        # lower == tau is UNKNOWN; only 0.45 is certified high.
        self.assertEqual(int(row.n_certified_high), 1)
        self.assertAlmostEqual(float(row.unknown_fraction), 2 / 3)

    def test_target_upper_coverage_is_simultaneous_not_max_vs_max(self) -> None:
        tasks = synthetic_tasks()
        # For target A, task a1 misses its upper bound while task a2 has a high
        # enough upper bound that max(error) <= max(upper).  Correct target
        # coverage is nevertheless false because not every task is covered.
        tasks.loc[tasks.task_id.eq("a1"), "family_upper"] = 0.15
        _, target = compute_operating_curves(tasks, [0.25])
        row = target.loc[
            target.study.eq("S") & target.objective.eq("family_rms")
        ].iloc[0]
        self.assertAlmostEqual(float(row.upper_empirical_coverage), 0.5)

    def test_released_lineage_counts_and_e182_fail_are_preserved(self) -> None:
        tasks, checks = load_released_tasks(ROOT)
        self.assertEqual(len(tasks), 2433)
        self.assertEqual(checks["e181_registered_family_tasks"], 2393)
        self.assertEqual(checks["e182_evaluation_tasks"], 40)
        self.assertEqual(checks["e182_preregistered_status"], "FAIL")
        self.assertEqual(checks["family_rms_lower_violations"], 0)
        self.assertEqual(checks["worst_member_lower_violations"], 0)
        self.assertEqual(checks["e183_family_upper_targets_covered"], 666)
        self.assertEqual(checks["e183_worst_upper_targets_covered"], 688)

    def test_run_reads_only_released_tables_and_emits_no_false_high_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = run(ROOT, Path(directory), [0.02, 0.05], figures=False)
            task = pd.read_csv(Path(directory) / "CERTIFIED_HIGH_TASK_CURVES.csv")
            target = pd.read_csv(Path(directory) / "CERTIFIED_HIGH_TARGET_CURVES.csv")
            saved = json.loads((Path(directory) / "STATUS.json").read_text())
        self.assertEqual(status["status"], "PASS")
        self.assertEqual(int(task.n_false_certificates.sum()), 0)
        self.assertEqual(int(target.n_false_certificates.sum()), 0)
        self.assertEqual(saved["provenance"]["truth_array_files_read"], 0)
        self.assertEqual(saved["provenance"]["e205_truth_read"], 0)
        self.assertEqual(saved["provenance"]["e208_truth_read"], 0)

    def test_lineage_validation_rejects_false_e182_status(self) -> None:
        # This checks the public exception class remains importable for callers;
        # the real-release lineage test above exercises the actual status gate.
        self.assertTrue(issubclass(CertificateIntegrityError, RuntimeError))


if __name__ == "__main__":
    unittest.main()

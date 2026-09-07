"""Drive the shipped paper-audit pack against official CSVs.

Run without extra deps::

    cd /home/yyf/proj
    PYTHONPATH=code/safeconf_audit python3 -m unittest code.safeconf_audit.tests.test_paper_pack
    PYTHONPATH=code/safeconf_audit python3 code/safeconf_audit/tests/test_paper_pack.py
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "safeconf_audit"))

from safeconf_audit.paper_pack import (  # noqa: E402
    E201_CORE,
    _row,
    check_documents,
    check_figures,
    check_gpt_mismatch,
    check_terms,
    load_claim_table,
    locked_quotes,
    pack_dir,
    round4,
    run_all_checks,
)

REPO = ROOT


class TestPaperPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = load_claim_table(REPO)

    def test_locked_display_equals_official_csv_rounding(self):
        risk = pd.read_csv(REPO / E201_CORE / "tables" / "E201_RISK_ASSOCIATIONS.csv")
        partial = pd.read_csv(REPO / E201_CORE / "tables" / "E201_PARTIAL_ASSOCIATIONS.csv")
        utility = pd.read_csv(REPO / E201_CORE / "tables" / "E201_REVIEW_UTILITY.csv")
        error = pd.read_csv(REPO / E201_CORE / "tables" / "E201_TARGET_ERROR_SUMMARY.csv")
        locked = self.table["locked_display"]

        sc = _row(risk, scope="pooled", predictor="safeconf_e201_risk")
        mag = _row(risk, scope="pooled", predictor="predicted_magnitude")
        part = _row(partial, scope="pooled", predictor="safeconf_e201_risk")
        u_sc = _row(utility, scope="pooled", predictor="safeconf_e201_risk")
        u_mag = _row(utility, scope="pooled", predictor="predicted_magnitude")
        k562_err = _row(
            error, stratum="primary_ge30", target="K562", predictor="four_seed_family"
        )

        self.assertEqual(locked["safeconf_pooled_spearman"], round4(sc.estimate))
        self.assertEqual(locked["magnitude_pooled_spearman"], round4(mag.estimate))
        self.assertEqual(locked["partial_spearman"], round4(part.estimate))
        self.assertEqual(locked["safeconf_utility_20"], round4(u_sc.oracle_normalized_utility))
        self.assertEqual(locked["magnitude_utility_20"], round4(u_mag.oracle_normalized_utility))
        self.assertEqual(locked["n_primary_tasks"], str(int(sc.n_tasks)))
        self.assertEqual(locked["k562_centroid_rmse"], round4(k562_err.family_centroid_rmse_mean))
        self.assertEqual(locked["k562_control_rmse"], round4(k562_err.control_error_mean))
        self.assertEqual(
            locked["k562_official_rmse"],
            round4(k562_err.official_general_baseline_error_mean),
        )
        for scope, key in (
            ("K562", "k562_spearman"),
            ("RPE1", "rpe1_spearman"),
            ("hepg2", "hepg2_spearman"),
            ("jurkat", "jurkat_spearman"),
        ):
            row = _row(risk, scope=scope, predictor="safeconf_e201_risk")
            self.assertEqual(locked[key], round4(row.estimate))

    def test_e204_is_profile_only(self):
        self.assertTrue(self.table["e204"]["profile_only"])
        self.assertFalse(self.table["e204"]["formal_80_epoch_complete"])
        self.assertEqual(self.table["e205"]["n_result_tables"], 0)
        self.assertFalse(self.table["publication"]["q2_certain"])
        self.assertFalse(self.table["publication"]["q1_sprint_ready"])

    def test_documents_quote_locked_csv_values(self):
        missing = check_documents(REPO, self.table)
        self.assertEqual(missing, [])

    def test_teaching_glosses_required_terms(self):
        missing = check_terms(pack_dir(REPO) / "02_从零Nature图解教学.md")
        self.assertEqual(missing, [])

    def test_gpt_mismatch_classes_and_official_paths(self):
        missing = check_gpt_mismatch(pack_dir(REPO) / "01_完成度与GPT误导对照.md")
        self.assertEqual(missing, [])

    def test_figures_round_trip_official_csv(self):
        missing = check_figures(REPO, self.table)
        self.assertEqual(missing, [])

    def test_run_all_checks_pass(self):
        report = run_all_checks(REPO, self.table)
        self.assertTrue(report["ok"], json.dumps(report, ensure_ascii=False, indent=2))

    def test_claim_table_on_disk_matches_loader(self):
        on_disk = json.loads((pack_dir(REPO) / "CLAIM_TABLE.json").read_text())
        self.assertEqual(on_disk["locked_display"], locked_quotes(self.table))


if __name__ == "__main__":
    unittest.main()

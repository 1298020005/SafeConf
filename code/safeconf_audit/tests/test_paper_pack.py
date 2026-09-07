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
    check_fig1_source_only_weights,
    check_gpt_mismatch,
    check_terms,
    fig3_panel_b_label_errorbar_hits,
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

    def test_scratch_claim_check_lists_pass_per_locked_number(self):
        from tempfile import TemporaryDirectory
        from safeconf_audit.paper_pack import run_all_checks, write_scratch_reports

        report = run_all_checks(REPO, self.table)
        self.assertTrue(report["ok"])
        with TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            write_scratch_reports(REPO, scratch, self.table, report)
            text = (scratch / "claim_check.txt").read_text()
            for key, value in self.table["locked_display"].items():
                self.assertIn(f"[PASS] {key}={value}", text)
            self.assertIn("ALL PASS", text)
            terms = (scratch / "term_scan.txt").read_text()
            self.assertIn("[PASS] 扰动 (perturbation)", terms)
            self.assertIn("ALL PASS", terms)
            figs = (scratch / "figure_check.txt").read_text()
            self.assertIn("[PASS] fig3_e201_main panels a/b/c values match CSV rounding", figs)
            self.assertIn("[PASS] fig3b labels sit above errorbar whiskers", figs)
            self.assertIn("[PASS] fig1a training weights from 源域证据 not four seeds", figs)

    def test_fig3b_collision_detector_flags_label_on_whisker(self):
        svg = """<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
 <g id="axes_2">
  <g id="LineCollection_1">
   <path d="M 379.898275 170.597914
L 379.898275 138.345284
"/>
  </g>
  <text style="font-size: 7px" x="379.898275" y="145.561081">0.3200</text>
 </g>
</svg>
"""
        hits = fig3_panel_b_label_errorbar_hits(svg)
        self.assertTrue(hits, "detector must flag 0.3200 sitting on the whisker")
        self.assertTrue(any("0.3200" in row for row in hits))

    def test_fig3b_collision_detector_accepts_label_above_whisker(self):
        svg = """<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
 <g id="axes_2">
  <g id="LineCollection_1">
   <path d="M 379.898275 170.597914
L 379.898275 138.345284
"/>
  </g>
  <text style="font-size: 7px" x="379.898275" y="128.0">0.3200</text>
 </g>
</svg>
"""
        self.assertEqual(fig3_panel_b_label_errorbar_hits(svg), [])

    def test_component_spearman_matches_descriptive_csv(self):
        desc = pd.read_csv(
            REPO
            / E201_CORE
            / "tables"
            / "E201_DESCRIPTIVE_ASSOCIATIONS.csv"
        )
        disp = self.table["e201"]["component_display"]
        for key in disp:
            row = _row(
                desc,
                scope="pooled",
                stratum="primary_ge30",
                predictor=key,
                outcome="family_rms_error",
            )
            self.assertEqual(disp[key], round4(row.spearman))

    def test_journal_doc_names_target_venues_and_rejects_certain_q2(self):
        from safeconf_audit.paper_pack import check_journal_doc

        missing = check_journal_doc(REPO)
        self.assertEqual(missing, [])
        text = (pack_dir(REPO) / "05_期刊对照表.md").read_text()
        self.assertIn("Briefings in Bioinformatics", text)
        self.assertIn("Nature Methods", text)
        self.assertTrue((pack_dir(REPO) / "04_五成分与封存流程精讲.md").is_file())

    def test_component_teaching_doc_quotes_csv_components(self):
        from safeconf_audit.paper_pack import check_component_doc

        missing = check_component_doc(REPO, self.table)
        self.assertEqual(missing, [])

    def test_fig1a_training_weights_come_from_source_evidence(self):
        sidecar = json.loads(
            (pack_dir(REPO) / "figures" / "fig1_architecture.values.json").read_text()
        )
        svg = (pack_dir(REPO) / "figures" / "fig1_architecture.svg").read_text()
        missing = check_fig1_source_only_weights(sidecar["plotted"], svg)
        self.assertEqual(missing, [])
        edges = [tuple(item) for item in sidecar["plotted"]["layout"]["edges"]]
        self.assertIn(("source_evidence", "training_weights"), edges)
        self.assertNotIn(("four_seeds", "training_weights"), edges)
        self.assertNotIn(("safeconf", "training_weights"), edges)

    def test_journal_universe_covers_wos_lists_and_scores_each_title(self):
        from safeconf_audit.journal_universe import (
            BRM_OFFICIAL,
            EVIDENCE_HAVE as UNIVERSE_HAVE,
            MCB_OFFICIAL,
            catalog,
            check_universe_doc,
            counts,
            official_unmatched,
        )
        from safeconf_audit.paper_pack import EVIDENCE_HAVE

        self.assertEqual(UNIVERSE_HAVE, EVIDENCE_HAVE)
        self.assertEqual(len(MCB_OFFICIAL), 61)
        self.assertEqual(len(BRM_OFFICIAL), 82)
        rows = catalog()
        self.assertGreaterEqual(len(rows), 100)
        tally = counts(rows)
        self.assertEqual(tally["mcb_official"], 61)
        self.assertEqual(tally["brm_official"], 82)
        self.assertEqual(official_unmatched(rows), [])
        by_name = {row["name"].lower(): row for row in rows}
        self.assertEqual(by_name["chromatographia"]["verdict"], "off_track")
        self.assertEqual(by_name["bioinformatics"]["verdict"], "q2_not_ready")
        self.assertEqual(by_name["nature methods"]["verdict"], "q1_off")
        self.assertEqual(by_name["bmc bioinformatics"]["verdict"], "discuss_narrow")
        self.assertFalse(self.table["publication"]["q2_certain"])
        missing = check_universe_doc(REPO)
        self.assertEqual(missing, [])
        text = (pack_dir(REPO) / "06_全球相关期刊逐本评价.md").read_text()
        for row in rows:
            self.assertIn(row["name"], text)
        self.assertIn("不能把二区写成一定能发", text)
        csv_text = (pack_dir(REPO) / "journal_universe.csv").read_text()
        self.assertIn("Chromatographia", csv_text)
        self.assertIn("Bioinformatics", csv_text)


if __name__ == "__main__":
    unittest.main()

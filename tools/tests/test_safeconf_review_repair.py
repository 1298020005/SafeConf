"""Regression checks for the two confirmed review errors, without pytest."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts.run_safeconf_public_reliability_v1 import jackknife_mean
from tools.scripts.run_safeconf_source_gate_v1 import (
    fit_rank_reference, gate_score, transform_rank,
)
from tools.safeconf_continual.research import CDF_KEYS, shuffled_labels


class ReviewRepairTests(unittest.TestCase):
    def test_jackknife_equals_sample_variance_over_n_for_equal_weights(self):
        h = np.array([[0., 2.], [1., 3.], [2., 5.], [4., 8.]])
        _, j, status = jackknife_mean(h, np.ones(len(h)))
        expected = np.var(h, axis=0, ddof=1).mean() / len(h)
        self.assertEqual(status, 'ok')
        self.assertAlmostEqual(j * j, expected, places=12)

    def test_insufficient_history_is_not_zero_stability(self):
        _, j, _ = jackknife_mean(np.array([[0.], [1.]]), np.ones(2))
        self.assertTrue(np.isnan(j))

    def test_gate_is_invariant_to_positive_changes_of_channel_units(self):
        pfit, sfit = np.array([2., 4., 8.]), np.array([.1, .3, .8])
        p, s, bad = np.array([3., 5., 9.]), np.array([.2, .4, .9]), np.array([0.,1.,1.])
        a = gate_score(p,s,bad,.5,fit_rank_reference(pfit),fit_rank_reference(sfit))
        b = gate_score(p*100,s/100,bad,.5,fit_rank_reference(pfit*100),fit_rank_reference(sfit/100))
        np.testing.assert_allclose(a,b)
        reference = fit_rank_reference(pfit)
        before = reference.copy()
        transform_rank(np.array([-1.,1000.]),reference)
        np.testing.assert_array_equal(reference,before)

    def test_shuffle_preserves_context_labels_as_complete_gene_blocks(self):
        rows=[]
        for gene in ['a','b','c']:
            for target in ['context1','context2']:
                rows.append(dict(zip(CDF_KEYS,['study','output','source','version',target])) |
                            {'gene':gene,'task_id':f'{gene}_{target}'})
        frame=pd.DataFrame(rows)
        labels=np.array([.1,.2,.3,.4,.5,.6])
        permuted,audit=shuffled_labels(frame,labels,20260930)
        self.assertEqual(audit['moved_clusters'],3)
        original_blocks={tuple(labels[g.index]) for _,g in frame.groupby('gene')}
        for _,g in frame.groupby('gene'):
            self.assertIn(tuple(permuted[g.index]),original_blocks)
            self.assertFalse(np.array_equal(permuted[g.index],labels[g.index]))
        for _,g in frame.groupby('target'):
            np.testing.assert_array_equal(np.sort(labels[g.index]),np.sort(permuted[g.index]))


if __name__=='__main__':
    unittest.main()

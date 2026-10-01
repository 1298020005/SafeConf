"""Scientific invariants for budgeted risk learning (stdlib unittest)."""
import unittest
import numpy as np
import pandas as pd

from tools.safeconf_continual.research import (
    budget_subset, historical_distance, rank_labels, shuffled_labels, metrics,
    paired_prediction_wide,
)


def fixture():
    rows=[]
    for gene in range(30):
        for upstream in ['A','B']:
            for context in ['c1','c2']:
                rows.append({'gene':str(gene),'task_id':f'{gene}:{context}',
                             'upstream':upstream,'model_version':'1','target':context,
                             'dataset_id':'study','output_contract_id':'delta',
                             'true_error_rmse':gene+(.1 if upstream=='B' else 0)})
    return pd.DataFrame(rows)


class BudgetContractTests(unittest.TestCase):
    def test_pairing_survives_csv_precision_without_dropping_tasks(self):
        frame = pd.DataFrame({'task_id': ['a', 'b'], 'target': 'c', 'gene': ['g1', 'g2'],
                              'upstream': 'C', 'true_error_rmse': [.027122223334444455, .031234567891234567],
                              'method': 'A', 'risk': [.2, .8]})
        other = frame.copy(); other['method'] = 'B'
        other['true_error_rmse'] = np.nextafter(other.true_error_rmse, np.inf)
        got = paired_prediction_wide(pd.concat([frame, other], ignore_index=True))
        self.assertEqual(len(got), 2)
        self.assertFalse(got[['A', 'B']].isna().any().any())

    def test_pairing_rejects_different_truth_or_duplicate_prediction(self):
        a = pd.DataFrame({'task_id': ['a'], 'gene': ['g'], 'true_error_rmse': [.02],
                          'method': ['A'], 'risk': [.4]})
        b = a.copy(); b['method'] = 'B'; b['true_error_rmse'] = .03
        with self.assertRaisesRegex(ValueError, 'different task truths'):
            paired_prediction_wide(pd.concat([a, b]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            paired_prediction_wide(pd.concat([a, a]))

    def test_small_budget_cdf_cannot_read_excluded_errors(self):
        frame=fixture();small=budget_subset(frame,.1)
        labels,audit=rank_labels(small,'small',.1)
        changed=frame.copy()
        changed.loc[~changed.gene.isin(small.gene),'true_error_rmse']=1e9
        labels2,audit2=rank_labels(budget_subset(changed,.1),'small',.1)
        np.testing.assert_array_equal(labels,labels2)
        self.assertEqual(audit,audit2)
        self.assertTrue(all(r['n_rows']==3 for r in audit))

    def test_cdf_midrank_ties_and_resolution(self):
        frame=fixture().iloc[:4].copy()
        frame['upstream']='A';frame['target']='c1'
        frame['true_error_rmse']=[1,1,2,3]
        labels,audit=rank_labels(frame,'ties')
        np.testing.assert_array_equal(labels,[.25,.25,.625,.875])
        self.assertEqual(audit[0]['max_ecdf_jump'],.5)

    def test_block_shuffle_preserves_distribution_and_joint_layout(self):
        frame=fixture();labels,_=rank_labels(frame,'full')
        shuffled,audit=shuffled_labels(frame,labels,3)
        self.assertEqual(audit['moved_fraction'],1)
        self.assertFalse(np.array_equal(labels,shuffled))
        for _,group in frame.groupby(['upstream','target']):
            np.testing.assert_array_equal(np.sort(labels[group.index]),np.sort(shuffled[group.index]))
        for _,group in frame.groupby('gene'):
            # All four source/context rows retain the same latent difficulty.
            self.assertEqual(len(np.unique(shuffled[group.index])),1)

    def test_history_identity_and_single_source(self):
        p=np.array([2.,-1.]);h=np.array([[1.,0.],[3.,2.]])
        w=np.array([.25,.75])
        got=historical_distance(p,h,w)
        expected=np.sqrt(np.average(np.mean((h-p)**2,axis=1),weights=w))
        self.assertAlmostEqual(got,expected,places=12)
        self.assertAlmostEqual(historical_distance(p,h[:1],np.ones(1)),1.)
        with self.assertRaises(ValueError): historical_distance(p,h[:0],np.empty(0))

    def test_u20_ties_and_empty_denominator(self):
        frame=pd.DataFrame({'task_id':[f'{i:02d}' for i in range(20)],'true_error_rmse':np.arange(20)})
        self.assertEqual(metrics(frame,np.arange(20))['utility20'],1)
        self.assertTrue(np.isnan(metrics(frame.iloc[:19],np.arange(19))['utility20']))
        frame['true_error_rmse']=1
        self.assertTrue(np.isnan(metrics(frame,np.arange(20))['utility20']))


if __name__=='__main__': unittest.main()

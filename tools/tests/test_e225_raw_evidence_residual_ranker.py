import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

p=Path(__file__).parents[1]/'scripts/run_e225_raw_evidence_residual_ranker.py'
spec=importlib.util.spec_from_file_location('e225',p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestE225(unittest.TestCase):
    def test_ties_are_order_invariant(self):
        s=np.array([1.,1.,1.,.2,.1]); y=np.array([2.,5.,8.,1.,3.]); order=[2,0,4,1,3]
        self.assertAlmostEqual(m.metrics(s,y,.2)['utility'],m.metrics(s[order],y[order],.2)['utility'])
        np.testing.assert_allclose(m.top_weights(s,.2),[1/3,1/3,1/3,0,0])
    def test_budget_and_oracle(self):
        y=np.arange(1,12,dtype=float)
        self.assertAlmostEqual(m.top_weights(y,.2).sum(),3.)
        self.assertAlmostEqual(m.metrics(y,y,.2)['utility'],1.)
    def test_no_remaining_tasks_is_undefined_not_divide_by_zero(self):
        with np.errstate(all='raise'):
            r=m.metrics([.2],[.7],.2)
            self.assertTrue(np.isnan(r['utility']))
            self.assertTrue(np.isnan(r['remaining_relative_error']))
            self.assertAlmostEqual(r['capture'],1.)
    def test_error_and_calibrated_scores_not_features(self):
        self.assertNotIn(m.ERROR,m.FEATURES)
        self.assertNotIn('safeconf_calibrated_pair_risk',m.FEATURES)
    def test_equal_candidate_budgets(self):
        self.assertEqual(len(m.CANDIDATES),10)
        self.assertEqual(m.CANDIDATES[0][0],0.)
    def test_test_labels_cannot_change_prediction(self):
        n=16
        f=pd.DataFrame({'dataset':['A']*n,'fold_id':['f']*n,'train_fraction':[1.]*n,
            'setting':['random_missing_pair']*n,'m':np.linspace(.1,1,n),'d':np.linspace(1,.1,n),
            'cn':np.zeros(n),'scarcity':np.ones(n)*.2,m.ERROR:np.linspace(.2,.8,n)})
        altered=f.copy(); altered[m.ERROR]=999.
        a,_=m.fit_correction(f,f[m.FEATURES],'MDH_condition',10.)
        b,_=m.fit_correction(f,altered[m.FEATURES],'MDH_condition',10.)
        np.testing.assert_allclose(a,b)

if __name__=='__main__': unittest.main()

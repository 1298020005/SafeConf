import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
p=Path(__file__).parents[1]/'scripts/run_e231_history_quality.py'
s=importlib.util.spec_from_file_location('quality',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)


class TestQuality(unittest.TestCase):
    def test_control_labels(self):
        np.testing.assert_array_equal(m.parse_control(pd.Series([0,1,0])),[False,True,False])
        with self.assertRaises(ValueError): m.parse_control(pd.Series(['0','unknown']))
    def test_weights(self):
        for method in m.SOURCE_METHODS:
            w=m.precision_weights([1.,9.],[0.,2.],method)
            self.assertAlmostEqual(w.sum(),1.)
            self.assertTrue((w>=0).all())
        self.assertGreater(m.precision_weights([1.,9.],[0.,0.],'precision')[0],.5)
        with self.assertRaises(ValueError): m.precision_weights([-1.,1.],[0.,0.],'precision')
    def test_risk_fallback_and_no_labels(self):
        f=pd.DataFrame({'analysis_stratum':['primary_ge30']*4,'family_disagreement':[.3,.2,.1,.4],
            'model_source_gap':[.1,.2,.3,.4],'source_delta_dispersion':[.4,.3,np.nan,.1],
            'negative_log_source_cells':[-2.,-3.,-4.,0.],'log_mean_variance':[-3.,-2.,-1.,np.nan],
            'support_context_deficit':[0,1,2,3],'predicted_magnitude':[1.,2.,3.,4.],
            'n_source_contexts':[3,2,1,0],'reliability':[0.,.5,1.,1.]})
        a=m.risk_scores(f,True,True)
        self.assertEqual(a[0],.25); self.assertEqual(a[3],1.)
        f['family_rms_error']=[999.,0.,2.,3.]
        np.testing.assert_array_equal(a,m.risk_scores(f,True,True))
    def test_identical_uncertainty_weights_equal(self):
        np.testing.assert_allclose(m.precision_weights([1.,1.,1.],[0.,0.,0.],'combined'),np.ones(3)/3)


if __name__=='__main__': unittest.main()

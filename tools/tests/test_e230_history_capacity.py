import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
p=Path(__file__).parents[1]/'scripts/run_e230_history_capacity.py'
s=importlib.util.spec_from_file_location('e230',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

class TestHistoryCapacity(unittest.TestCase):
    def test_source_excludes_target(self):
        for target in m.original.TARGETS:
            for seed in m.SEEDS:
                order=m.ordered_sources(target,seed)
                self.assertNotIn(target,order); self.assertEqual(len(order),3)
    def test_nested_real_cell_sampling(self):
        x=m.subset_order(100,'c|p',11)
        self.assertEqual(len(set(x)),100)
        self.assertTrue(set(x[:25])<set(x[:50])<set(x))
        np.testing.assert_array_equal(x,m.subset_order(100,'c|p',11))
    def frame(self):
        return pd.DataFrame({'analysis_stratum':['primary_ge30']*4,'predicted_magnitude':[1.,2.,3.,4.],
            'family_disagreement':[.1,.4,.2,.3],'model_source_gap':[np.nan,.1,.2,.3],
            'source_delta_dispersion':[np.nan,np.nan,.1,.2],
            'negative_log_source_cells':[0,-2,-3,-4],'support_context_deficit':[3,2,1,0],
            'n_source_contexts':[0,1,2,3]})
    def test_missing_fallback_and_no_truth_dependency(self):
        f=self.frame(); a=m.score_features(f)
        f['family_rms_error']=[999.,0.,1.,2.]; b=m.score_features(f)
        np.testing.assert_allclose(a[list(m.SCORES)],b[list(m.SCORES)])
        self.assertEqual(a.m_plus_history.iloc[0],a.magnitude.iloc[0])
        self.assertTrue(np.isfinite(a[list(m.SCORES)]).all().all())
    def test_constant_component_is_safe(self):
        f=self.frame(); f['source_delta_dispersion']=np.nan
        self.assertTrue(np.isfinite(m.score_features(f)['history_risk']).all())

if __name__=='__main__': unittest.main()

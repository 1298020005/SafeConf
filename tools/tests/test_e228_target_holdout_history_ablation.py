import importlib.util
from pathlib import Path
import unittest,numpy as np,pandas as pd
p=Path(__file__).parents[1]/'scripts/run_e228_target_holdout_history_ablation.py'; s=importlib.util.spec_from_file_location('e228',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
class TestE228(unittest.TestCase):
 def test_tie_invariant(self):
  a=np.array([1.,1.,.2,.1]); y=np.array([4.,1.,3.,2.]); o=[2,0,3,1]
  self.assertEqual(m.metric(a,y,.2)['utility'],m.metric(a[o],y[o],.2)['utility'])
 def test_no_old_score(self): self.assertNotIn('safeconf_e201_risk',sum(m.RAW_FEATURES.values(),[]))
if __name__=='__main__': unittest.main()

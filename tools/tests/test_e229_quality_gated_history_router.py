import importlib.util
from pathlib import Path
import unittest
import numpy as np
p=Path(__file__).parents[1]/'scripts/run_e229_quality_gated_history_router.py'
s=importlib.util.spec_from_file_location('e229',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

class TestGate(unittest.TestCase):
    def test_branch_and_scale(self):
        out=m.quality_gate([.01,.02,.03],[.9,.3,.6],[1,3,3],[1,0,1])
        np.testing.assert_allclose(out,[1/3,1/3,1])
    def test_monotone_rescaling_preserves_output(self):
        a=m.quality_gate([.01,.02,.03],[.9,.3,.6],[1,3,3],[1,0,0])
        b=m.quality_gate([100,200,300],[90,30,60],[1,3,3],[1,0,0])
        np.testing.assert_allclose(a,b)
    def test_bad_data_fail(self):
        with self.assertRaises(ValueError): m.quality_gate([1,np.nan],[2,3],[1,3],[0,0])
        with self.assertRaises(ValueError): m.quality_gate([1],[2],[4],[0])

if __name__=='__main__': unittest.main()

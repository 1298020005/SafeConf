import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
p=Path(__file__).parents[1]/'scripts/run_e227_local_nonlinear_ranker.py'
s=importlib.util.spec_from_file_location('e227',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

class TestE227(unittest.TestCase):
    def frame(self):
        n=64
        return pd.DataFrame({'dataset':['A']*n,'fold_id':['f']*n,'train_fraction':[1.]*n,
            'setting':['random_missing_pair']*n,'m':np.linspace(.01,1,n),
            'd':np.linspace(1,.01,n),'cn':np.zeros(n),'scarcity':np.ones(n)*.2,
            m.core.ERROR:np.linspace(.1,.8,n)})
    def test_design_dimensions(self):
        for arm,n in zip(m.core.ARMS,(1,2,4,8)):
            self.assertEqual(m.inputs(self.frame(),arm).shape,(64,n))
    def test_inference_does_not_use_target_error(self):
        a=self.frame(); b=a.copy(); b[m.core.ERROR]=999
        for family in m.FAMILIES:
            with m.threadpool_limits(limits=1):
                x=m.predict(a,a[m.core.FEATURES],family,'MDH_condition')
                y=m.predict(a,b[m.core.FEATURES],family,'MDH_condition')
            np.testing.assert_allclose(x,y)

if __name__=='__main__': unittest.main()

import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
p=Path(__file__).resolve().parents[1]/'scripts/run_safeconf_model_decision.py'
spec=importlib.util.spec_from_file_location('decision',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class LeakageTests(unittest.TestCase):
    def test_final_test_labels_are_not_parsed_into_numeric_frame(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.csv'
            p.write_text('fold_id,split,true_error_rmse\n0,test,SEALED_NONNUMERIC\n0,val,0.5\n1,val,NOT_ALLOWED\n0,train,IN_SAMPLE\n')
            x=m.allowed_frame(p,['fold_id','split','true_error_rmse'])
            self.assertEqual(x.true_error_rmse.to_numpy(float).tolist(),[.5])
    def test_query_truth_cannot_change_features_or_anchor(self):
        fit=pd.DataFrame({'prediction_l2_norm':[1.,2.,3.,4.], 'true_error_rmse':[1.,1.,2.,4.]})
        query=pd.DataFrame({'prediction_l2_norm':[2.,np.nan], 'true_error_rmse':[1.,2.]})
        x,z=m.features(fit,query,m.P[:1]); a=m.anchor_predict(fit,query)
        query['true_error_rmse']=[1e9,-1e9]
        xx,zz=m.features(fit,query,m.P[:1]); aa=m.anchor_predict(fit,query)
        np.testing.assert_array_equal(x,xx); np.testing.assert_array_equal(z,zz); np.testing.assert_array_equal(a,aa)
    def test_anchor_oof_excludes_own_perturbation_labels(self):
        fit=pd.DataFrame({'perturbation':np.repeat(list('abcdef'),2), 'prediction_l2_norm':np.arange(12)+1.,'true_error_rmse':np.arange(12)*.1+.2})
        before=m.anchor_oof(fit)
        fit.loc[fit.perturbation=='a','true_error_rmse']=1e8
        after=m.anchor_oof(fit)
        np.testing.assert_allclose(before[:2],after[:2],rtol=0,atol=0)
    def test_tied_risk_has_random_expected_utility(self):
        self.assertAlmostEqual(m.utility(np.ones(10),np.arange(10.)),0.)
        self.assertAlmostEqual(m.utility(np.arange(10.),np.arange(10.)),1.)

if __name__=='__main__': unittest.main()

"""Numerical checks of biological bootstrap and evaluation-only endpoints."""
import numpy as np
import pandas as pd
from tools.safeconf_continual.submission_evidence import ClusterBootstrap, point_metrics, endpoint_arrays


def test_count_bootstrap_matches_explicit_cluster_copies_with_score_ties():
    rng=np.random.default_rng(41)
    f=pd.DataFrame([{'gene':f'g{i:02d}','task_id':f'g{i:02d}::{c}','target':c}
                    for i in range(24) for c in ['A','B']])
    e=rng.uniform(.01,.2,len(f));s=rng.integers(0,4,len(f)).astype(float)
    boot=ClusterBootstrap(f,e,replicates=31,seed=99)
    draws=np.random.default_rng(99).integers(0,24,(31,24))
    observed=boot.utility(s)
    explicit=[]
    genes=sorted(f.gene.unique())
    for draw in draws:
        indices=np.concatenate([np.flatnonzero(f.gene.eq(genes[j])) for j in draw])
        q=f.iloc[indices].reset_index(drop=True)
        values=[]
        for rows in q.groupby('target').indices.values():
            ii=indices[np.asarray(rows)]
            values.append(point_metrics(e[ii],s[ii],f.task_id.to_numpy()[ii])['utility'])
        explicit.append(np.mean(values))
    np.testing.assert_allclose(observed,explicit,rtol=1e-12,atol=1e-12)


def test_top200_is_actual_truth_effect_order_not_prediction_order():
    truth=np.arange(300,dtype=float)[None,:]
    pred=truth.copy();pred[0,0]=1000
    e=endpoint_arrays(pred,truth)
    assert e['delta_rmse'][0]>0
    assert e['effect_top200_rmse'][0]==0
    constant=np.ones_like(truth)
    assert np.isnan(endpoint_arrays(constant,truth)['pearson_error'][0])


def test_fixed_twenty_percent_severe_set_at_other_review_budgets():
    ids=np.asarray([f'q{i:02d}' for i in range(40)])
    error=np.arange(40,dtype=float)
    m=point_metrics(error,error,ids,.05)
    assert m['review_k']==2 and m['severe_k']==8 and m['high_error_found']==2
    assert m['high_error_precision']==1 and m['high_error_recall']==.25


def test_external_gate_rejects_constant_mean_even_with_zero_relative_gap():
    from tools.scripts.run_safeconf_gladstone_v21 import competence_gate
    rng=np.random.default_rng(71)
    tasks=pd.DataFrame([{'gene':f'g{i:02d}','context':c} for i in range(60) for c in ['Rest','Stim8hr','Stim48hr']])
    truth=rng.normal(size=(len(tasks),24))
    mean=np.zeros_like(truth)
    constant=competence_gate(tasks,mean,truth,mean)
    assert constant['relative_gap']==0 and constant['status']=='FAIL'
    accurate=competence_gate(tasks,.99*truth,truth,mean)
    assert accurate['status']=='PASS' and accurate['ci95'][1]<=.02

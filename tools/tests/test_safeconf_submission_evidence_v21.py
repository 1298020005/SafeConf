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


def test_frozen_score_interface_missing_history_and_truth_isolation():
    from tools.safeconf_continual.frozen_scoring import score_task
    cfg={'method':'PublicRule','version':'frozen-test','channel_cdfs':{'Amplitude':[1.,2.,3.],'Public':[.1,.2,.3]}}
    task={'predicted_magnitude':2.,'public_available':False,'public_raw':np.nan}
    score,state,version=score_task(task,cfg)
    assert score==.5 and state=='NO_HISTORY_AMPLITUDE_FALLBACK' and version=='frozen-test'
    assert score_task(dict(task,true_error_rmse=999.,query_truth=-99.),cfg)==(score,state,version)
    legal=dict(task,public_available=True,public_raw=.3)
    assert score_task(legal,cfg)[0]==5/6
    supervised={'method':'FrozenSupervised','version':'source-only'}
    assert score_task({'frozen_model_score':.7,'true_error_rmse':-99.},supervised)[0]==.7


def test_coalesced_range_boundaries_reuse_and_unexpected_full_response():
    import tempfile,threading,json
    from pathlib import Path
    from unittest.mock import Mock
    from tools.safeconf_continual.range_h5 import MeteredHTTPFile
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);reader=object.__new__(MeteredHTTPFile)
        reader.cache=p;reader.block=8;reader.size=24;reader.reserved=0;reader.cap=100000
        reader.ledger=p/'ledger.json';reader.sessions=threading.local();reader.url='fake'
        response=Mock();response.status_code=206;response.headers={'Content-Range':'bytes 0-15/24'}
        response.content=b'abcdefghijklmnop';response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        reader.sessions.session=Mock();reader.sessions.session.get.return_value=response
        reader.prefetch_blocks([0,1])
        assert (p/'00000000.bin').read_bytes()==b'abcdefgh' and (p/'00000001.bin').read_bytes()==b'ijklmnop'
        assert json.loads(reader.ledger.read_text())['this_run_network_payload_bytes']==16
        reader.prefetch_blocks([0,1]);assert reader.sessions.session.get.call_count==1
        assert reader.sessions.session.get.call_args.kwargs['stream'] is True
        response.status_code=200
        try:reader.prefetch_blocks([2])
        except RuntimeError as err:assert 'Range response mismatch' in str(err)
        else:raise AssertionError('must refuse a full-object response to a Range request')


def test_saved_predictor_batch_reload_and_truth_independence():
    import tempfile
    from pathlib import Path
    import torch
    from tools.scripts import run_safeconf_gladstone_v21 as m
    torch.set_num_threads(4);rng=np.random.default_rng(57);original=m.OUT
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m.OUT=Path(tmp);genes=np.asarray([f'g{i:03d}' for i in range(37)])
            np.savez(m.OUT/'CONTROL_FEATURES.npz',genes=genes,embedding=rng.normal(0,6,(37,50)).astype(np.float32))
            (m.OUT/'OUTPUT_CONTRACT.json').write_text('{"n_output_genes":64}')
            for c in m.CONTEXTS:
                np.savez(m.OUT/f'RIDGE_{c}.npz',coef=rng.normal(size=(64,50)).astype(np.float32),
                    intercept=np.zeros(64,np.float32),mean=np.ones(64,np.float32),
                    x_mean=np.zeros(50,np.float32),x_scale=np.full(50,2.,np.float32))
                for seed in m.MODEL_SEEDS:
                    torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(50,256),torch.nn.ReLU(),torch.nn.Linear(256,64))
                    torch.save(net.state_dict(),m.OUT/f'MLP_{c}_{seed}.pt')
            tasks=pd.DataFrame([{'gene':g,'context':c} for g in [*genes,'unknown'] for c in m.CONTEXTS])
            for name in ['ridge','mlp']:
                bulk=m.frozen_predict(name,tasks)
                ix=np.asarray([0,3,6,17,20,41,88,111,112,113]);subset=tasks.iloc[ix]
                split=m.frozen_predict(name,subset)
                np.testing.assert_allclose(split,bulk[ix],rtol=1e-5,atol=1e-7)
                np.testing.assert_array_equal(split,m.frozen_predict(name,subset.assign(query_truth=999.,true_error_rmse=-8.)))
                np.testing.assert_array_equal(bulk[-3:],np.ones((3,64),np.float32))
            try:m.frozen_predict('mlp',pd.DataFrame([{'gene':'g001','context':'unregistered'}]))
            except ValueError:pass
            else:raise AssertionError('unknown context must not return uninitialized outputs')
        finally:m.OUT=original


def test_csc_reader_gathers_only_allowed_count_bytes():
    import tempfile,h5py
    from pathlib import Path
    from scipy.sparse import csc_matrix
    from tools.scripts.run_safeconf_kolf_panel_v21 import permitted_csc_column
    matrix=np.arange(1,46,dtype=np.float32).reshape(9,5)
    role_codes=np.asarray([0,1,2,-1,-1,-1,-1,-1,-1])
    with tempfile.TemporaryDirectory() as tmp:
        for poison in [False,True]:
            values=matrix.copy()
            if poison:values[3:]+=10000
            sparse=csc_matrix(values);path=Path(tmp)/f'counts_{poison}.h5'
            with h5py.File(path,'w') as h:
                g=h.create_group('counts');g.create_dataset('data',data=sparse.data,chunks=(8,))
                g.create_dataset('indices',data=sparse.indices.astype(np.int64),chunks=(8,))
                g.create_dataset('indptr',data=sparse.indptr.astype(np.int64))
            with h5py.File(path,'r') as h,path.open('rb') as raw:
                for column in range(5):
                    rows,observed=permitted_csc_column(raw,h['counts'],sparse.indptr,column,role_codes)
                    np.testing.assert_array_equal(rows,np.arange(3))
                    np.testing.assert_array_equal(observed,matrix[:3,column])

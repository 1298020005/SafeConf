"""Generated fixtures only. No actual Orion raw root is accepted."""
from pathlib import Path
import importlib.util,json,struct,sys,uuid
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
REPO=Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921');sys.path.insert(0,str(REPO))
from tools.scripts import evaluate_safeconf_orion_guarded_test_agent as t
b=t.biology;r=t.risk_api


def immutable(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n');path.chmod(0o444)
def bind(path):
    return {'path':str(Path(path).resolve()),'sha256':t.sha(path),'bytes':Path(path).stat().st_size}
def ledger(root,tsv=False):
    items=[{'path':p.name,'bytes':p.stat().st_size,'sha256':t.sha(p)} for p in sorted(root.iterdir()) if p.is_file() and not p.name.startswith('ARTIFACT_HASHES')]
    path=root/('ARTIFACT_HASHES.tsv' if tsv else 'ARTIFACT_HASHES.json')
    if tsv:pd.DataFrame(items).to_csv(path,sep='\t',index=False)
    else:immutable(path,items)
    return path
def table(path,frame):frame.to_csv(path,sep='\t',index=False,float_format='%.17g')


def fixture(base,bad=None):
    md=base/'metadata';raw=base/'raw';source=base/'source';fit=base/'fit';risk=base/'risk';comparison=base/'comparison'
    for folder in [md,raw,source,fit,risk,comparison]:folder.mkdir(parents=True)
    immutable(md/'SYNTHETIC_FIXTURE_ONLY.json',{'synthetic_only':True})
    full=pd.DataFrame({'ensembl_id':['E0','E1','E2'],'gene_name':['G1','G2','G3'],'gene_token_id':[0,1,2]})
    pq.write_table(pa.Table.from_pandas(full,preserve_index=False),md/'gene_metadata.parquet')
    endpoint=pd.DataFrame({'axis_index':[0,1],'source_index':[0,2],'gene_name':['G1','G3'],'orion_gene_token_id':[0,2],'orion_ensembl_id':['E0','E2']})
    endpoint.to_csv(md/'GENE_MANIFEST.csv',index=False)
    pd.DataFrame({'gene':['G1','G2','G3'],'gene_role':['TEST','TEST','VALIDATION']}).to_csv(md/'GENE_SPLIT.csv',index=False)
    immutable(md/'PREPARATION_POLICY.json',{'synthetic_only':True,'normalization_scale':4000})
    identities=[];objects=[];all_queries=[];contexts=[];fit_models=[];fit_registry={};lm_bindings=[]
    sentinel=918273645.125;token_sentinel=918273645
    for context in t.CONTEXTS:
        prefix=f'SyntheticStudy|';units=[f'{prefix}G1|{context}']*30+[f'{prefix}G2|{context}']*29+[f'{prefix}G3|{context}']*3
        roles=['TEST']*59+['TRAIN','VALIDATION','DEV_EXPOSED_CELL'];genes=['G1']*30+['G2']*29+['G3']*3
        ids=[[0,2] for _ in range(59)]+[[token_sentinel] for _ in range(3)]
        values=[[1.,3.] for _ in range(59)]+[[sentinel] for _ in range(3)]
        if context=='HCT116' and bad=='sum':values[0]=[1.,2.]
        if context=='HCT116' and bad=='unknown':ids[0]=[0,3]
        if context=='HCT116' and bad=='fractional':values[0]=[1.5,2.5]
        path=raw/f'{context}_Batch1.parquet'
        pq.write_table(pa.table({'gene_token_id':pa.array(ids,type=pa.list_(pa.int64())),'gene_expression':pa.array(values,type=pa.list_(pa.float64()))}),path,compression='snappy',use_dictionary=True,data_page_version='2.0',row_group_size=62)
        metadata=md/f'{context}_rows.parquet'
        pq.write_table(pa.table({'original_row_index':list(range(62)),'original_row_group':[0]*62,'row_role':roles,'gene_target':genes,
            'sample':[context+'_Batch1']*62,'cell_barcode':[f'{context}:{i}' for i in range(62)],'total_counts':[4.]*59+[1.]*3,
            'biological_unit':units,'fixed_metadata_QC_pass':[True]*62}),metadata)
        metadata.chmod(0o444)
        identities.append({'path':path.name,'context':context,'sample':context+'_Batch1','rows':62,'row_groups':1,
            'publisher_whole_file_bytes':path.stat().st_size,'publisher_LFS_sha256':t.sha(path),'metadata_parquet_path':str(metadata),'metadata_parquet_sha256':t.sha(metadata)})
        objects.append({'path':path.name,'bytes':path.stat().st_size,'lfs_sha256':t.sha(path)})
        folder=fit/context;folder.mkdir()
        queries=pd.DataFrame([[f'{prefix}G1|{context}','E0','G1',context,'TEST'],[f'{prefix}G2|{context}','E1','G2',context,'TEST'],[f'{prefix}G3|{context}','E2','G3',context,'VALIDATION']],columns=r.QUERY)
        all_queries.append(queries);table(folder/'QUERY_IDENTITIES.tsv',queries)
        axis=pd.DataFrame({'gene_id':['E0','E2'],'gene_symbol':['G1','G3']});table(folder/'OUTPUT_GENE_AXIS.tsv',axis)
        predicted=pd.DataFrame({'gene_id':['E0','E2'],**{q:[.5,.5] for q in queries.query_id}})
        predicted.to_csv(folder/'PREDICTIONS_DELTA.tsv.gz',sep='\t',index=False,compression='gzip')
        control=pd.DataFrame({'gene_id':['E0','E2'],'mean_cell_logCP4000':[.2,.3] if context=='HCT116' else [.5,.1]});table(folder/'OWN_CONTEXT_NTC_MEAN.tsv',control)
        (folder/'MODEL.rds').write_bytes(b'IMMUTABLE_SYNTHETIC_FULL_FIT_MODEL_'+context.encode())
        (folder/'SIMPLE_BASELINES.rds').write_bytes(b'IMMUTABLE_SYNTHETIC_BASELINE')
        table(folder/'BASELINE_SELECTION.tsv',pd.DataFrame({'selected':['zero_effect']}))
        status={'status':'PASS','mode':'fit','context_id':context,'pca_dim':'10','ridge_penalty':'0.1','seed':'1',
                'query_truth_read':'FALSE','input_estimand':'mean_cell_log1p_cp4000_v1','endpoint_gene_count':'3285','query_count':'3'}
        table(folder/'STATUS.tsv',pd.DataFrame({'key':list(status),'value':list(status.values())}))
        fit_models.append({'context_id':context,'model_sha256':t.sha(folder/'MODEL.rds')})
        fit_registry[context]=bind(ledger(folder,True))
    table(fit/'RUN_MANIFEST.tsv',pd.DataFrame(fit_models))
    for context in t.CONTEXTS:
        contexts.append(r.context_bindings(fit,fit,context,fit))
    immutable(md/'FILE_METADATA_IDENTITY.json',identities)
    manifest=base/'OBJECTS.json';immutable(manifest,{'release_sha':'SYNTHETIC','files':objects})
    science=base/'SCIENCE.json';immutable(science,{'synthetic_only':True,'normalization':{'formula':'mean_cell(log1p(4000 * raw_gene_UMI / official_full_library_total_counts))','unknown_token_policy':'fail_closed'}})
    endpoint.to_csv(source/'GENE_MANIFEST.csv',index=False)
    immutable(source/'GENE_IDS.json',['G1','G3']);immutable(source/'SOURCE_ERROR_CDF.json',{'scope':'HISTORICAL_SOURCE_ONLY'})
    immutable(source/'SOURCE_FEATURE_MANIFEST.json',{'risk_P_columns':r.P,'risk_public_columns':r.PUBLIC,'public_pair_columns':r.PAIR})
    models=[]
    for ref,kind in r.METHODS:
        name=f'{ref}_{kind}.joblib';(source/name).write_bytes(('FROZEN_SOURCE_ONLY_'+name).encode());columns=r.P if ref=='P_only' else r.P+r.PUBLIC
        models.append({'role':'Source_risk','reference':ref,'learner':kind,'path':name,'sha256':t.sha(source/name),'columns':columns,
            'target':'Source_per_upstream_per_context_midrank_CDF','numeric_model_width_with_missing_flags':2*len(columns)})
    name='PUBLIC.joblib';(source/name).write_bytes(b'FROZEN_SOURCE_RAW_RMSE')
    models.append({'role':'Public_biology_retrieval','path':name,'sha256':t.sha(source/name),'columns':r.PAIR,'target':'biological_transfer_rmse',
        'predict_clip':False,'numeric_model_width_with_missing_flags':18})
    immutable(source/'MODEL_MANIFEST.json',{'models':models})
    source_bindings=[bind(p) for p in sorted(source.iterdir()) if p.is_file()]
    role_registry=base/'ROLES.csv'
    pd.DataFrame([{'context':c,'task_range':'all gene roles TEST','role':'SEEN','metadata_seen':True,'result_seen':False,
        'test_truth_access_status':'CLOSED_UNTOUCHED_LABELS','pristine_confirmation_claim':False} for c in t.CONTEXTS]).to_csv(role_registry,index=False)
    queries=pd.concat(all_queries,ignore_index=True)
    comparison_manifest=base/'RISK_COMPARISON.json'
    immutable(comparison_manifest,{'schema':r.SCHEMA,'pretruth_inference_authorized':True,'target_errors_or_CDF_used':False,
        'risk_code_sha256':t.sha(r.__file__),'source_core_bindings':source_bindings,'scientific_contract_sha256':t.sha(science),'contexts':contexts,
        'data_role_registry_sha256':t.sha(role_registry)})
    _,_,_,lm_bindings=r.load_inputs(json.loads(comparison_manifest.read_text()),{'axis':endpoint})
    risk_manifest=risk/'RISK_SEAL_MANIFEST.json'
    immutable(risk_manifest,{'status':'SEALED_PRETRUTH','scientific_contract_sha256':t.sha(science),'source_only_risk_models':6,
        'query_errors_used':False,'target_CDF_fit':False,'new_learners_or_parameter_fits':0,'query_truth_input_or_dummy_labels':False,
        'risk_code_sha256':t.sha(r.__file__),'source_core_bindings':source_bindings,'LM_prediction_and_control_bindings':lm_bindings,
        'comparison_manifest_sha256':t.sha(comparison_manifest),'TEST_truth_status':'CLOSED','TEST_identity_metadata_status':'SEEN_METADATA_ONLY',
        'data_role_registry_sha256':t.sha(role_registry)})
    scores=queries.copy()
    for column in t.RISK_COLUMNS:scores[column]=.2
    scores['source_history_n']=1;table(risk/'PRETRUTH_RISKS.tsv',scores)
    pq.write_table(pa.table({'synthetic_placeholder':[1]}),risk/'P_ONLY_FEATURES.parquet')
    pq.write_table(pa.table({'synthetic_placeholder':[1]}),risk/'SOURCE_HISTORY_PAIR_FEATURES.parquet')
    table(risk/'SOURCE_PRIOR_WEIGHTS.tsv',pd.DataFrame({'synthetic':[1]}))
    for ref in ['Uniform','Manual','Learned']:
        pq.write_table(pa.table({'synthetic_placeholder':[1]}),risk/f'{ref}_P_PUBLIC_FEATURES.parquet')
        np.save(risk/f'{ref}_PRIOR_EFFECTS.npy',np.zeros((6,2)),allow_pickle=False)
    risk_registry=ledger(risk)
    scores.to_parquet(comparison/'PRETRUTH_COMPARISON_SCORES.parquet',index=False);(comparison/'PRETRUTH_COMPARISON_SCORES.parquet').chmod(0o444)
    spec=comparison/'COMPARISON_SPEC.json';immutable(spec,{'scientific_contract_sha256':t.sha(science),'risk_seal_manifest_sha256':t.sha(risk_manifest),
        'scores_sha256':t.sha(comparison/'PRETRUTH_COMPARISON_SCORES.parquet'),'test_truth_or_errors_used':False,'new_fits':0,'candidate_score_columns':t.RISK_COLUMNS,
        'evaluator_code_sha256':t.sha(REPO/'tools/scripts/evaluate_safeconf_orion_frozen_risk_agent.py')})
    competence=base/'COMPETENCE_RESULT.json';immutable(competence,{'status':'PASS','passes_competence':True,'actual_context_count':2,'noninferior_contexts':2,
        'noninferior_strata_fraction':1.0,'stable_disadvantage':False,'relative_macro_error_gap':0.,'eligible_validation_tasks':2,'evaluated_validation_tasks':2,
        'model_macro_rmse':1.,'baseline_macro_rmse':1.,'lower95_relative_gap':0.,'upper95_relative_gap':0.,
        'missing_predictions':0,'nonfinite_predictions':0,'test_truth_read':False,'scientific_contract_sha256':t.sha(science),'bootstrap_replicates':5000,
        'bootstrap_seed':20260929,'new_model_parameters_fitted':False,'synthetic_only':True})
    operation=base/'TEST_ACCESS.json';output=base/'evaluated'
    implementation={'test_reader':Path(t.__file__),'numeric_backend':Path(t.reader.__file__),'structural_reader':Path(t.reader.v1.__file__),
        'scientific_loader':Path(b.__file__),'risk_sealer':Path(r.__file__)}
    receipt={'schema':t.SCHEMA,'postseal_TEST_operation_authorized':True,'test_numeric_materialization_permitted':True,'authorized_numeric_roles':['TEST'],
        'authorized_contexts':list(t.CONTEXTS),'TEST_metadata_before_open':'SEEN','TEST_truth_before_open':'CLOSED','synthetic_only':True,
        'scientific_contract':bind(science),'forty_object_manifest':bind(manifest),'metadata_root':str(md),'raw_root':str(raw),
        'metadata_bindings':{name:t.sha(md/name) for name in t.META_NAMES},'data_role_registry':bind(role_registry),'final_competence':bind(competence),
        'lm_fit_root':str(fit),'lm_fit_run_manifest':bind(fit/'RUN_MANIFEST.tsv'),'lm_fit_artifact_registries':fit_registry,
        'risk_seal_root':str(risk),'risk_seal_manifest':bind(risk_manifest),'risk_seal_artifacts':bind(risk_registry),'risk_comparison_manifest':bind(comparison_manifest),
        'pretruth_comparison':{'spec':bind(spec),'scores':bind(comparison/'PRETRUTH_COMPARISON_SCORES.parquet')},
        'implementation_bindings':{key:bind(path) for key,path in implementation.items()},'test_semantic_QC':t.SEMANTIC_QC,'task_min_cells':30,
        'normalization_scale':4000,'output_path':str(output)}
    immutable(operation,receipt)
    return operation,output,((struct.pack('<q',token_sentinel),),(struct.pack('<d',sentinel),))


def run():
    code_sha=t.sha(t.__file__)
    base=t.DOC/'synthetic'/uuid.uuid4().hex;base.mkdir(parents=True)
    op,output,sentinels=fixture(base/'positive')
    result=t.evaluate_test(op,output,sentinels)
    errors=pd.read_parquet(output/'TEST_TASK_ERRORS.parquet');effects=np.load(output/'TEST_ENDPOINT_EFFECTS.npy')
    assert result['status']=='COMPLETE' and len(errors)==2 and errors.n_cells.eq(30).all()
    expected_means=np.log1p(np.asarray([1000.,3000.]))
    for i,row in enumerate(errors.itertuples(index=False)):
        control=np.asarray([.2,.3] if row.context_id=='HCT116' else [.5,.1]);expected=expected_means-control
        np.testing.assert_allclose(effects[i],expected,rtol=0,atol=1e-14)
        assert np.isclose(row.true_error_rmse,np.sqrt(np.mean((.5-expected)**2)))
    qc=pd.read_csv(output/'TEST_TASK_QC.tsv',sep='\t')
    assert len(qc)==4 and sorted(qc.n_cells)==[29,29,30,30]
    assert result['TEST_numeric_cells_opened']==118
    assert all(not p.stat().st_mode&0o222 for p in output.iterdir())
    cases=[{'case':'complete_exact_TEST_mask_private_sentinels_CP4000_endpoint_TRAINcontrols_postopen30_QC','passed':True}]
    for label,mutation in [('absent_authorization',lambda x:x.__setitem__('postseal_TEST_operation_authorized',False)),
        ('wrong_reader_SHA',lambda x:x['implementation_bindings']['test_reader'].__setitem__('sha256','0'*64)),
        ('missing_candidate_seal',lambda x:x['risk_seal_manifest'].__setitem__('sha256','0'*64)),
        ('unfrozen_comparison',lambda x:x['pretruth_comparison']['spec'].__setitem__('sha256','0'*64)),
        ('altered_semantic_QC',lambda x:x['test_semantic_QC'].__setitem__('library_sum_rtol',1e-6))]:
        op,out,s=fixture(base/label);receipt=json.loads(op.read_text());mutation(receipt);op.chmod(0o600);immutable(op,receipt)
        calls=[];original=t.reader.iter_selected_numeric_lists
        def tracked(*args,**kwargs):calls.append('numeric');yield from original(*args,**kwargs)
        t.reader.iter_selected_numeric_lists=tracked
        try:
            try:t.evaluate_test(op,out,s);raise AssertionError('Invalid gate accepted')
            except RuntimeError:assert not out.exists() and not calls
        finally:t.reader.iter_selected_numeric_lists=original
        cases.append({'case':label,'passed':True,'rejected_before_numeric_reader':True})
    for label,mutate in [('failed_competence',lambda c:c.__setitem__('status','FAIL_COMPETENCE')),
            ('faulty_competence_numeric_evidence',lambda c:c.__setitem__('model_macro_rmse',float('nan'))),
            ('TEST_truth_already_seen',None)]:
        op,out,s=fixture(base/label);receipt=json.loads(op.read_text())
        if mutate is not None:
            path=Path(receipt['final_competence']['path']);value=json.loads(path.read_text());mutate(value);path.chmod(0o600);immutable(path,value)
            receipt['final_competence']=bind(path)
        else:
            path=Path(receipt['data_role_registry']['path']);frame=pd.read_csv(path);frame['test_truth_access_status']='OPEN';frame.to_csv(path,index=False);receipt['data_role_registry']=bind(path)
        op.chmod(0o600);immutable(op,receipt)
        calls=[];original=t.reader.iter_selected_numeric_lists
        def tracked(*args,**kwargs):calls.append('numeric');yield from original(*args,**kwargs)
        t.reader.iter_selected_numeric_lists=tracked
        try:
            try:t.evaluate_test(op,out,s);raise AssertionError('Bad prerequisite accepted')
            except RuntimeError:assert not out.exists() and not calls
        finally:t.reader.iter_selected_numeric_lists=original
        cases.append({'case':label,'passed':True,'rejected_before_numeric_reader':True})
    op,out,s=fixture(base/'exact_competence_margin');receipt=json.loads(op.read_text());path=Path(receipt['final_competence']['path'])
    value=json.loads(path.read_text());value.update(model_macro_rmse=1.02,relative_macro_error_gap=1.02-1.,lower95_relative_gap=1.02-1.,upper95_relative_gap=1.02-1.)
    path.chmod(0o600);immutable(path,value);receipt['final_competence']=bind(path);op.chmod(0o600);immutable(op,receipt)
    t.authorize_test(op)
    cases.append({'case':'exact_competence_margin_preserves_original_ratio_comparison_no_epsilon','passed':True})
    for label in ['missing_final_competence','changed_full_LM_parameters','missing_candidate_prior_artifact']:
        op,out,s=fixture(base/label);receipt=json.loads(op.read_text())
        if label=='missing_final_competence':Path(receipt['final_competence']['path']).unlink()
        elif label=='changed_full_LM_parameters':
            path=Path(receipt['lm_fit_root'])/'HCT116/MODEL.rds';value=bytearray(path.read_bytes());value[0]^=1;path.write_bytes(value)
        else:(Path(receipt['risk_seal_root'])/'Learned_PRIOR_EFFECTS.npy').unlink()
        calls=[];iterator=t.reader.iter_selected_numeric_lists;checksum=t.sha
        def tracked(*args,**kwargs):calls.append('numeric_reader');yield from iterator(*args,**kwargs)
        def tracked_sha(path):
            if Path(path).resolve().is_relative_to(Path(receipt['raw_root']).resolve()):calls.append('raw_checksum')
            return checksum(path)
        t.reader.iter_selected_numeric_lists=tracked;t.sha=tracked_sha
        try:
            try:t.evaluate_test(op,out,s);raise AssertionError('Missing/faulty frozen prerequisite accepted')
            except RuntimeError:assert not out.exists() and not calls,calls
        finally:t.reader.iter_selected_numeric_lists=iterator;t.sha=checksum
        cases.append({'case':label,'passed':True,'rejected_before_raw_checksum_and_numeric_reader':True})
    for kind in ['sum','unknown','fractional']:
        op,out,s=fixture(base/kind,kind)
        try:t.evaluate_test(op,out,s);raise AssertionError('Bad TEST semantics accepted')
        except RuntimeError:
            assert not out.exists()
            stages=list(out.parent.glob('evaluated.incomplete.*'));assert len(stages)==1
            assert (stages[0]/'ABORTED_TEST_OPEN.json').is_file() and not (stages[0]/'TEST_TASK_ERRORS.parquet').exists()
        cases.append({'case':kind+'_aborts_entire_output_after_legal_open','passed':True})
    assert t.sha(t.__file__)==code_sha
    report={'status':'SYNTHETIC_GUARDED_TEST_READER_PASS_REAL_TEST_UNOPENED','test_reader_sha256':code_sha,
        'actual_TEST_numeric_access':False,'cases':cases,'generated_fixture_root':str(base),'new_fits':0,'target_CDF_fit':False}
    if (t.DOC/'SYNTHETIC_PROOF.json').exists():(t.DOC/'SYNTHETIC_PROOF.json').chmod(0o600)
    immutable(t.DOC/'SYNTHETIC_PROOF.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()

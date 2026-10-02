#!/usr/bin/env python3
"""Source-only matched-support physical history-content null, fixed B1/B2 arms.

Effect content is reassigned from each actual fit-only bank within context and
fit-derived log-cell deciles. Metadata, controls, support and group membership
remain physical. Frozen biology parameters score recomputed inputs; risk models
are fitted again with the identical rows, labels, CDF and HGB configuration.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from tools.scripts import run_safeconf_publicset_risk_followup_v1 as follow
from tools.scripts import run_safeconf_publicset_v1 as pub

OUT = follow.OUT/'registered_universal_v1'/'content_null_v1'
RUNTIME = follow.RUNTIME/'registered_universal_v1'/'content_null_v1'
REAL_OUT = follow.OUT/'registered_universal_v1'
REAL_RISK = follow.RUNTIME/'registered_universal_v1'/'risk'
BUILDERS = ('B1_HGB','B2_Pointwise')
ORDERS = (0,1,2,3,4)


def key_hash(*values):
    return hashlib.sha256('|'.join(map(str,values)).encode()).hexdigest()


def validate_support(real, changed):
    if len(real) != len(changed): raise RuntimeError('null group coverage changed')
    for a,b in zip(real,changed):
        if a['q']!=b['q'] or not np.array_equal(a['ix'],b['ix']) or not np.array_equal(a['s'],b['s']):
            raise RuntimeError('null membership/support changed')
        metadata = [i for i,c in enumerate(pub.SOURCE_FIELDS) if c not in
                    ('source_effect_magnitude','source_effect_abs_mean','source_conflict')]
        if not np.array_equal(a['x'][:,metadata],b['x'][:,metadata],equal_nan=True):
            raise RuntimeError('null metadata/control feature changed')


def shuffled_domain(d, fitrows, queryrows, outer, inner, order):
    realfit, realquery = d.groups(fitrows,set()), d.groups(queryrows,set())
    bank = follow.validate_groups(d,realfit,d.tasks.iloc[queryrows].gene.unique())
    if not set(d.memory.iloc[bank].perturbation_target)<=set(d.tasks.iloc[fitrows].gene):
        raise RuntimeError('physical fit bank contains a non-fitting biological gene')
    needed = np.unique(np.concatenate([g['ix'] for g in realfit+realquery if len(g['ix'])]))
    memory = d.memory
    mapping = np.arange(len(memory)); deciles = np.full(len(memory),-1,int)
    bins = {}; blockrows=[]
    for context in sorted(memory.iloc[bank].context.astype(str).unique()):
        donors = bank[memory.iloc[bank].context.astype(str).to_numpy()==context]
        cells = memory.iloc[donors].n_cells.to_numpy(float)
        ok = np.isfinite(cells)&(cells>0)
        donors = donors[ok]
        if not len(donors): continue
        logcells = np.log1p(memory.iloc[donors].n_cells.to_numpy(float))
        cuts = np.unique(np.quantile(logcells,np.arange(1,10)/10))
        eligible = needed[memory.iloc[needed].context.astype(str).to_numpy()==context]
        ecells = memory.iloc[eligible].n_cells.to_numpy(float)
        valid = np.isfinite(ecells)&(ecells>0)
        eligible = eligible[valid]; ecells=ecells[valid]
        deciles[eligible] = np.searchsorted(cuts,np.log1p(ecells),side='right')
        for decile in sorted(set(deciles[donors])):
            group = donors[deciles[donors]==decile]
            ordered = sorted(group,key=lambda i:key_hash('PublicSet-content-null-v1',outer,inner,order,
                                                         memory.iloc[i].experiment_id))
            ordered=np.asarray(ordered,int)
            if len(ordered)>1: mapping[ordered]=np.roll(ordered,1)
            bins[(context,int(decile))]=ordered
            blockrows.append({'context':context,'log_cells_decile':int(decile),'n_fit_donors':len(ordered),
                              'cycle_permutable':len(ordered)>1})
    inbank = np.zeros(len(memory),bool);inbank[bank]=True
    queryhist = np.unique(np.concatenate([g['ix'] for g in realquery if len(g['ix'])]))
    unknown=[]
    for recipient in queryhist:
        if inbank[recipient]: continue
        context=str(memory.iloc[recipient].context); decile=deciles[recipient]
        donors=bins.get((context,int(decile)))
        if donors is None or not len(donors):
            unknown.append(int(recipient)); continue
        slot=int(key_hash('PublicSet-content-null-query-v1',outer,inner,order,
                          memory.iloc[recipient].experiment_id),16)%len(donors)
        mapping[recipient]=donors[slot]
    moved=needed[mapping[needed]!=needed]
    known=needed[~np.isin(needed,unknown)]
    if not set(mapping[moved])<=set(bank): raise RuntimeError('null effect donor outside actual fit bank')
    prohibited=set(d.tasks.iloc[queryrows].gene)
    if set(memory.iloc[mapping[moved]].perturbation_target)&prohibited:
        raise RuntimeError('null query gene used as donor')
    for g in realfit+realquery:
        if np.any(memory.iloc[mapping[g['ix']]].context.astype(str).to_numpy()==str(d.tasks.iloc[g['q']].context)):
            raise RuntimeError('current context treated query experiment entered null history')
    effects=d.effects.copy();effects[needed]=d.effects[mapping[needed]]
    nd=replace(d,effects=effects)
    nf,nq=nd.groups(fitrows,set()),nd.groups(queryrows,set())
    validate_support(realfit,nf);validate_support(realquery,nq)
    table=memory.iloc[needed][['experiment_id','perturbation_target','context','n_cells']].copy()
    table['effect_donor_id']=memory.iloc[mapping[needed]].experiment_id.to_numpy()
    table['donor_gene']=memory.iloc[mapping[needed]].perturbation_target.to_numpy()
    table['log_cells_decile']=deciles[needed]
    table['recipient_in_fit_bank']=inbank[needed]
    table['effect_changed']=np.any(d.effects[needed]!=effects[needed],axis=1)
    table['identity_changed']=mapping[needed]!=needed
    table['unknown_no_matched_fit_donor']=np.isin(needed,unknown)
    table['n_donors_in_matching_bin']=[len(bins.get((str(memory.iloc[i].context),int(deciles[i])),[])) for i in needed]
    audit={'outer':outer,'inner':inner,'null_order':order,'fit_bank_rows':len(bank),
        'fit_bank_ids_hash':pub.fingerprint(memory.iloc[bank].experiment_id),
        'fit_gene_hash':pub.fingerprint(d.tasks.iloc[fitrows].gene.unique()),
        'query_gene_hash':pub.fingerprint(prohibited),'fit_query_ids_hash':pub.fingerprint(d.tasks.iloc[fitrows].task_id),
        'query_ids_hash':pub.fingerprint(d.tasks.iloc[queryrows].task_id),
        'all_changed_content_donors_fit_only':True,'all_fit_bank_genes_actual_fit_queries':True,'query_gene_donors':0,
        'current_context_treated_query_experiments_used':0,
        'metadata_controls_support_exactly_preserved':True,'n_needed_rows':len(needed),
        'n_identity_changed':len(moved),'effective_identity_shuffle_rate':len(moved)/len(needed),
        'effective_content_shuffle_rate':float(table.effect_changed.mean()),
        'n_unknown_no_match':len(unknown),'n_singleton_fit_bins':sum(x['n_fit_donors']==1 for x in blockrows),
        'known_matched_fraction':len(known)/len(needed),'matching':'same physical context and fit-only log1p(n_cells) decile',
        'effect_features_conflicts_weights_priors_discrepancies_recomputed':True,
        'unknown_policy':'Retain legal original recipient content only when no matched fit donor exists; mark unknown, never fabricate',
        'MC_TEST_reads':0}
    return nd,nf,nq,table,audit,pd.DataFrame(blockrows)


def freeze_b1(d,fitgroups,outer,inner):
    import joblib
    if inner is not None:
        source=follow.RUNTIME/f'nested/outer{outer}/inner{inner}/B1_HGB/seed0/fit'
        prep,model=joblib.load(source/'MODEL.joblib')
        return prep,model,{'source_model_binding':follow.binding(source/'MODEL.joblib'),'new_biology_fit':0}
    # Original outer B1 runner exported pair arrays and scores, not its model.
    # Reproduce those real fits once before freezing; no null labels are used.
    destination=RUNTIME/f'frozen_real_b1/outer{outer}'
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'MODEL.joblib').exists():
        prep,model=joblib.load(destination/'MODEL.joblib')
        return prep,model,follow.read_json(destination/'FIT_AUDIT.json')
    from tools.safeconf_continual.learners import NumericPreprocessor
    from sklearn.ensemble import HistGradientBoostingRegressor
    source=None
    for run in ('full_v2_resume','full_v1','prototype_v2_cpu_environment_repair'):
        candidate=pub.RUNTIME/f'{run}/Source/outer{outer}/B1_HGB'
        if (candidate/'HGB_INPUT.npz').exists(): source=candidate;break
    if source is None: raise RuntimeError('outer real B1 source arrays missing')
    with np.load(source/'HGB_INPUT.npz',allow_pickle=False) as z:
        x,y,q=[z[c].copy() for c in ('fit_x','fit_y','query_x')]
    actual=np.concatenate([g['x'] for g in fitgroups])
    sourceaudit=follow.read_json(source/'FIT_AUDIT.json')
    if sourceaudit['fit_query_ids_hash']!=pub.fingerprint(d.tasks.iloc[[g['q'] for g in fitgroups]].task_id):
        raise RuntimeError('B1 real exported fitting IDs mismatch')
    # Registered NumPy environments differ in log1p by up to one float64 ULP.
    if not np.allclose(x,actual,rtol=1e-12,atol=1e-14,equal_nan=True):
        raise RuntimeError('B1 real exported fit features materially changed')
    prep=NumericPreprocessor().fit(x)
    model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_depth=3,min_samples_leaf=20,
        l2_regularization=10.,random_state=pub.SEEDS[0]).fit(prep.transform(x),y)
    old=np.load(source/'SCORES.npy');reproduced=model.predict(prep.transform(q))
    maxdiff=float(np.max(np.abs(old-reproduced)))
    if not np.allclose(old,reproduced,rtol=1e-10,atol=1e-12): raise RuntimeError('B1 outer real fit failed exact reproduction')
    joblib.dump((prep,model),destination/'MODEL.joblib')
    audit={'source_input_binding':follow.binding(source/'HGB_INPUT.npz'),'source_score_binding':follow.binding(source/'SCORES.npy'),
        'reproduced_max_absolute_score_difference':maxdiff,
        'real_fit_features_max_absolute_difference':float(np.nanmax(np.abs(x-actual))),
        'real_fit_task_ids_exactly_matched':True,'new_biology_fit':1,'null_content_biology_training_rows':0,
        'model_binding':follow.binding(destination/'MODEL.joblib')}
    follow.stage_json(destination/'FIT_AUDIT.json',audit)
    return prep,model,audit


def priors(args,neural=False):
    d=follow.source_domain();torch.set_num_threads(4)
    device=torch.device(f'cuda:{args.gpu}' if neural else 'cpu')
    builder='B2_Pointwise' if neural else 'B1_HGB'
    phase='b2_frozen_inference' if neural else 'b1_frozen_inference'
    start=time.monotonic();cap=args.gpu_hours_cap*3600 if neural else args.cpu_minutes_cap*60
    status={'status':'RUNNING','pid':os.getpid(),'phase':phase,'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
            'fixed_orders':args.orders,'fixed_seeds':list(pub.SEEDS),'MC_TEST_reads':0}
    follow.stage_json(OUT/f'{phase.upper()}_STATUS.json',status)
    for outer in args.folds:
        outerfit,outerquery,_=follow.split_rows(d,outer)
        for inner in list(range(4))+[None]:
            fitrows,queryrows=(outerfit,outerquery) if inner is None else follow.split_rows(d,outer,inner)[:2]
            realfit=d.groups(fitrows,set())
            if not neural: prep,model,fitaudit=freeze_b1(d,realfit,outer,inner)
            for order in args.orders:
                tag='outer' if inner is None else f'inner{inner}'
                base=RUNTIME/f'order{order}/outer{outer}/{tag}'
                seeds=pub.SEEDS if neural else (0,)
                if all((base/builder/f'seed{s}/PRIORS.npz').exists() for s in seeds):continue
                if time.monotonic()-start>=cap: raise TimeoutError('physical null budget reached before next inference stage')
                nd,nf,nq,mapping,audit,bins=shuffled_domain(d,fitrows,queryrows,outer,inner,order)
                base.mkdir(parents=True,exist_ok=True)
                pub.atomic_csv(base/'DONOR_MAPPING.csv.gz',mapping);pub.atomic_csv(base/'MATCHING_BINS.csv',bins)
                audit.update(donor_mapping_binding=follow.binding(base/'DONOR_MAPPING.csv.gz'),
                             matching_bins_binding=follow.binding(base/'MATCHING_BINS.csv'))
                follow.stage_json(base/'SHUFFLE_AUDIT.json',audit)
                for seed in seeds:
                    destination=base/builder/f'seed{seed}'
                    if (destination/'PRIORS.npz').exists():continue
                    if neural:
                        source=(follow.lookup_outer_model(outer,builder,seed) if inner is None else
                                follow.RUNTIME/f'nested/outer{outer}/inner{inner}/{builder}/seed{seed}/refit')
                        model,prep,fitaudit=pub.reuse_model(nd,nf,source,device)
                        weights=pub.predict_weights(model,prep,nd,nq,device)
                        reuse={'model_binding':follow.binding(source/'MODEL.pt'),
                               'preprocessor_binding':follow.binding(source/'PREPROCESSOR.joblib'),'new_biology_fit':0,
                               'frozen_real_pca_and_numeric_transform':True,'effect_codes_recomputed':True}
                    else:
                        weights={}
                        for g in nq:
                            score=model.predict(prep.transform(g['x']))
                            w=np.exp(np.clip(-(score-score.min())/max(float(score.std()),1e-8),-20,20));w/=w.sum()
                            weights[g['q']]=.5*w+.5*g['s']
                        reuse=fitaudit|{'new_null_biology_fit':0}
                    follow.save_priors(nd,nq,weights,destination,audit|reuse|{'builder':builder,'public_seed':seed,
                        'control':'physical history-content null','fit_error_labels':0,
                        'biology_parameters':'Frozen real fit; transformed effect features and weights recomputed'})
                    print(json.dumps({'completed_null_prior':str(destination.relative_to(RUNTIME)),
                                      'effective_content_shuffle_rate':audit['effective_content_shuffle_rate']}),flush=True)
                    if neural:del model,prep;torch.cuda.empty_cache()
    records=[follow.read_json(p)|{'path':str(p)} for p in sorted(RUNTIME.glob('order*/outer*/*/SHUFFLE_AUDIT.json'))]
    pub.atomic_csv(OUT/'SHUFFLE_LEDGER.csv',pd.DataFrame(records))
    follow.stage_json(OUT/f'{phase.upper()}_STATUS.json',status|{'status':'COMPLETE','elapsed_seconds':time.monotonic()-start,
        'new_null_biology_models':0,'actual_inference_priors':len(list(RUNTIME.glob(f'order*/outer*/*/{builder}/seed*/PRIORS.npz')))})


def merged(outer,builder,seed,order):
    publicseed=seed if builder=='B2_Pointwise' else 0
    pieces=[follow.read_priors(RUNTIME/f'order{order}/outer{outer}/inner{inner}/{builder}/seed{publicseed}') for inner in range(4)]
    data={k:np.concatenate([p[k] for p in pieces]) for k in ('qrows','task_ids','priors','summary')}
    orderrows=np.argsort(data['qrows'])
    if len(np.unique(data['qrows']))!=len(data['qrows']):raise RuntimeError('null nested OOF duplicate')
    return {k:v[orderrows] for k,v in data.items()}


def cpu(args):
    import joblib
    from tools.safeconf_continual.research import P,PUBLIC,rank_labels,fit_risk
    d=follow.source_domain();start=time.monotonic();waiting=0.
    status={'status':'RUNNING','pid':os.getpid(),'phase':'null_cpu_risk','orders':args.orders,'builders':args.builders,
            'MC_TEST_reads':0,'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    statuspath=OUT/('CPU_'+'_'.join(args.builders)+'_STATUS.json')
    follow.stage_json(statuspath,status)
    for outer in args.folds:
        fit,query,_=follow.split_rows(d,outer)
        for order in args.orders:
            for builder in args.builders:
                for seed in pub.SEEDS:
                    publicseed=seed if builder=='B2_Pointwise' else 0
                    required=[RUNTIME/f'order{order}/outer{outer}/inner{i}/{builder}/seed{publicseed}/PRIORS.npz' for i in range(4)]
                    required.append(RUNTIME/f'order{order}/outer{outer}/outer/{builder}/seed{publicseed}/PRIORS.npz')
                    realpublicseed=seed if builder=='B2_Pointwise' else 0
                    realriskseed=seed if builder=='B2_Pointwise' else pub.SEEDS[0]
                    required.extend(REAL_RISK/f'{source}_to_{target}/{context}/outer{outer}/{builder}/publicseed{realpublicseed}/riskseed{realriskseed}/TRAINING_ERRORS_CDF.csv.gz'
                        for source,target in [('TxPert_GAT','TxPert_Exphormer'),('TxPert_Exphormer','TxPert_GAT')] for context in follow.CONTEXTS)
                    while any(not p.exists() for p in required):
                        if not args.wait_for_inputs:raise FileNotFoundError('Required real/null inputs incomplete; rerun with --wait-for-inputs')
                        follow.stage_json(statuspath,status|{'status':'WAITING_FOR_FIXED_INPUTS','waiting_seconds':waiting,
                            'missing_paths':[str(p) for p in required if not p.exists()],
                            'active_elapsed_seconds':time.monotonic()-start-waiting})
                        waitstart=time.monotonic();time.sleep(5);waiting+=time.monotonic()-waitstart
                    trprior=merged(outer,builder,seed,order)
                    quprior=follow.read_priors(RUNTIME/f'order{order}/outer{outer}/outer/{builder}/seed{publicseed}')
                    if set(trprior['qrows'])!=set(fit) or set(quprior['qrows'])!=set(query):
                        raise RuntimeError('null/real outer rows differ')
                    for source,target in [('TxPert_GAT','TxPert_Exphormer'),('TxPert_Exphormer','TxPert_GAT')]:
                        trframe=follow.feature_frame(d,trprior,source);quframe=follow.feature_frame(d,quprior,target)
                        for context in follow.CONTEXTS:
                            train=trframe[trframe.target==context].reset_index(drop=True)
                            test=quframe[quframe.target==context].reset_index(drop=True)
                            path=RUNTIME/f'risk/order{order}/{source}_to_{target}/{context}/outer{outer}/{builder}/seed{seed}'
                            if (path/'PREDICTIONS.csv.gz').exists():continue
                            if time.monotonic()-start-waiting>=args.cpu_minutes_cap*60:
                                raise TimeoutError('null risk CPU wall budget reached')
                            path.mkdir(parents=True,exist_ok=True);run_id=str(path.relative_to(RUNTIME))
                            labels,cdfa=rank_labels(train,run_id)
                            realpublicseed=seed if builder=='B2_Pointwise' else 0
                            realriskseed=seed if builder=='B2_Pointwise' else pub.SEEDS[0]
                            realpath=REAL_RISK/f'{source}_to_{target}/{context}/outer{outer}/{builder}/publicseed{realpublicseed}/riskseed{realriskseed}'
                            realledger=pd.read_csv(realpath/'TRAINING_ERRORS_CDF.csv.gz').set_index('task_id')
                            if set(realledger.index)!=set(train.task_id) or not np.allclose(realledger.loc[train.task_id].training_rank_label,labels,atol=1e-14):
                                raise RuntimeError('null/real CDF training rows or rank labels differ')
                            fitstart=time.monotonic();model=fit_risk(train,labels,P+PUBLIC,'hgb',seed)
                            risks=model.predict(test);elapsed=time.monotonic()-fitstart
                            joblib.dump(model,path/'MODEL.joblib')
                            ledger=train[['dataset_id','output_contract_id','upstream','model_version','target','task_id','gene','fold','true_error_rmse']].copy()
                            ledger['training_rank_label']=labels;pub.atomic_csv(path/'TRAINING_ERRORS_CDF.csv.gz',ledger)
                            follow.stage_json(path/'CDF_AUDIT.json',cdfa)
                            result=test[['task_id','gene','target','fold','upstream','true_error_rmse']].copy()
                            result['risk']=risks;result['method']=builder+'_ContentNull';result['public_seed']=seed
                            result['risk_seed']=seed;result['source_upstream']=source;result['null_order']=order
                            pub.atomic_csv(path/'PREDICTIONS.csv.gz',result)
                            follow.stage_json(path/'FIT_AUDIT.json',{'scope':run_id,'method':builder+'_ContentNull','builder':builder,
                                'public_seed':seed,'risk_seed':seed,'null_order':order,'source_upstream':source,'target_upstream':target,
                                'context':context,'outer':outer,'n_train':len(train),'n_test':len(test),
                                'feature_columns':P+PUBLIC,'real_training_error_binding':follow.binding(realpath/'TRAINING_ERRORS_CDF.csv.gz'),
                                'identical_real_training_rows_and_CDF_labels':True,'target_error_fit_rows':0,'MC_TEST_reads':0,
                                'elapsed_seconds':elapsed,'model_binding':follow.binding(path/'MODEL.joblib'),
                                'prediction_binding':follow.binding(path/'PREDICTIONS.csv.gz')})
                            print(json.dumps({'completed_null_risk':run_id,'fit_seconds':elapsed}),flush=True)
    aggregate()
    follow.stage_json(statuspath,status|{'status':'COMPLETE_REQUESTED_ARMS',
        'elapsed_seconds':time.monotonic()-start,'active_elapsed_seconds':time.monotonic()-start-waiting,'waiting_seconds':waiting})


def aggregate():
    import fcntl
    with open(RUNTIME/'AGGREGATE.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return aggregate_locked()


def aggregate_locked():
    from tools.safeconf_continual.research import metrics
    frames=[];fits=[];cdfs=[]
    for path in sorted((RUNTIME/'risk').glob('order*/*/*/outer*/*/seed*/FIT_AUDIT.json')):
        fits.append(follow.read_json(path));frames.append(pd.read_csv(path.parent/'PREDICTIONS.csv.gz'))
        cdfs.extend(follow.read_json(path.parent/'CDF_AUDIT.json'))
    if not frames:return
    null=pd.concat(frames,ignore_index=True)
    real=pd.read_csv(REAL_OUT/'TASK_PREDICTIONS.csv.gz')
    real=real[real.method.isin(['Support_only']+list(BUILDERS))]
    orders=sorted(null.null_order.unique())
    paired=pd.concat([null]+[real.assign(null_order=o) for o in orders],ignore_index=True)
    keys=['source_upstream','upstream','target','fold','method','public_seed','risk_seed','null_order']
    rows=[dict(zip(keys,key))|metrics(p,p.risk.to_numpy()) for key,p in paired.groupby(keys,sort=True)]
    metricscols=['utility20','spearman','aurc','high_risk_miss_rate','error_at_10','error_at_20','error_at_50']
    fold=pd.DataFrame(rows)
    macro=fold.groupby(['source_upstream','upstream','method','public_seed','risk_seed','null_order'],as_index=False)[metricscols].mean()
    pub.atomic_csv(OUT/'TASK_PREDICTIONS.csv.gz',paired)
    pub.atomic_csv(OUT/'CONTEXT_FOLD_RESULTS.csv',fold)
    pub.atomic_csv(OUT/'MACRO_RESULTS.csv',macro)
    pub.atomic_csv(OUT/'SEED_ORDER_MEAN_MACRO_RESULTS.csv',macro.groupby(['source_upstream','upstream','method'],as_index=False)[metricscols].mean())
    pub.atomic_csv(OUT/'RISK_FIT_LEDGER.csv',pd.DataFrame(fits));pub.atomic_csv(OUT/'CDF_LEDGER.csv',pd.DataFrame(cdfs))
    follow.stage_json(OUT/'RESULT_MANIFEST.json',{'status':'COMPLETE' if len(fits)==1200 else 'PARTIAL',
        'actual_risk_fits':len(fits),'expected_risk_fits':1200,'null_task_prediction_rows':len(null),'expected_null_rows':1808*2*3*5*2,
        'fixed_builders':list(BUILDERS),'fixed_seeds':list(pub.SEEDS),'fixed_null_orders':list(ORDERS),
        'readiness':'All five arms required for registered paired statistics','MC_TEST_reads':0,
        'biology_models':'Frozen real parameters; B1 outer exported fits reproduced once; no null biology refitting',
        'primary':'20 context by outer-fold macro, metric mean across3seeds and5orders',
        'bindings':[follow.binding(OUT/x) for x in ('TASK_PREDICTIONS.csv.gz','CONTEXT_FOLD_RESULTS.csv','MACRO_RESULTS.csv','RISK_FIT_LEDGER.csv','CDF_LEDGER.csv')]})


def statistics(args):
    while args.wait_for_inputs:
        completed=len(list((RUNTIME/'risk').glob('order*/*/*/outer*/*/seed*/FIT_AUDIT.json')))
        if completed==1200:break
        follow.stage_json(OUT/'STATISTICS_STATUS.json',{'status':'WAITING_FOR_ALL_FIXED_ARMS','pid':os.getpid(),
            'actual_risk_fits':completed,'expected_risk_fits':1200,'MC_TEST_reads':0})
        time.sleep(5)
    start=time.monotonic()
    aggregate()
    frame=pd.read_csv(OUT/'TASK_PREDICTIONS.csv.gz')
    methods=['Support_only','B1_HGB','B2_Pointwise','B1_HGB_ContentNull','B2_Pointwise_ContentNull']
    contrasts=[]
    for builder in BUILDERS:
        contrasts.extend([(builder,'Support_only'),(builder,builder+'_ContentNull'),(builder+'_ContentNull','Support_only')])
    follow.paired_statistics(frame,OUT,RUNTIME,methods,contrasts,args.bootstrap,ORDERS,'null_order')
    follow.stage_json(OUT/'STATISTICS_STATUS.json',{'status':'COMPLETE','pid':os.getpid(),
        'active_elapsed_seconds':time.monotonic()-start,'MC_TEST_reads':0})
    resource_receipt()


def resource_receipt():
    receipts={}
    for name in ('B1_FROZEN_INFERENCE_STATUS','B2_FROZEN_INFERENCE_STATUS',
                 'CPU_B1_HGB_STATUS','CPU_B2_Pointwise_STATUS','STATISTICS_STATUS'):
        receipts[name]=follow.read_json(OUT/(name+'.json'))
    # Charge the interrupted first B1 generation attempt as well as its durable
    # resumed stage. Waiting for a prerequisite cache is not compute phase time.
    attempts=follow.read_json(OUT/'SERVICE_LIFECYCLE_RECEIPT.json')
    failed_seconds=0.
    for item in attempts['units']:
        if item['unit']=='safeconf-publicset-content-null-b1-v1.service':
            failed_seconds=(int(item['ExecMainExitTimestampMonotonic'])-
                            int(item['ExecMainStartTimestampMonotonic']))/1e6
    cpu_seconds=(failed_seconds+receipts['B1_FROZEN_INFERENCE_STATUS']['elapsed_seconds']+
        receipts['CPU_B1_HGB_STATUS']['active_elapsed_seconds']+
        receipts['CPU_B2_Pointwise_STATUS']['active_elapsed_seconds']+
        receipts['STATISTICS_STATUS']['active_elapsed_seconds'])
    gpu_seconds=receipts['B2_FROZEN_INFERENCE_STATUS']['elapsed_seconds']
    follow.stage_json(OUT/'RESOURCE_BUDGET_RECEIPT.json',{
        'status':'COMPLETE','accounting':'Compute phase wall time; prerequisite-cache waiting excluded and separately recorded',
        'CPU_compute_phase_seconds':cpu_seconds,'CPU_budget_seconds':1800,'CPU_within_budget':cpu_seconds<=1800,
        'GPU_frozen_inference_phase_seconds':gpu_seconds,'GPU_budget_seconds':1800,'GPU_within_budget':gpu_seconds<=1800,
        'charged_interrupted_B1_prior_attempt_seconds':failed_seconds,
        'outer_real_B1_reconstruction_fits':len(list(RUNTIME.glob('frozen_real_b1/outer*/FIT_AUDIT.json'))),
        'outer_real_B1_reconstruction_cost':'Included in B1 frozen inference phase, never null biology training',
        'actual_null_risk_fits':len(list((RUNTIME/'risk').glob('order*/*/*/outer*/*/seed*/FIT_AUDIT.json'))),
        'new_null_biology_models':0,'frozen_neural_models_reused':75,'MC_TEST_reads':0,
        'phase_receipts':receipts,
        'implementation_bindings':[follow.binding(__file__),follow.binding(follow.__file__),follow.binding(pub.__file__)]})


def register():
    OUT.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(parents=True,exist_ok=True)
    if (OUT/'REGISTRATION.json').exists():return
    follow.stage_json(OUT/'REGISTRATION.json',{'role':'NEW_DEV_SEEN_MATCHED_SUPPORT_PHYSICAL_CONTENT_NULL',
        'scope':'Source1808tasks/575genes/2840genes; five outer and four inner folds; no MC reads',
        'builders':list(BUILDERS),'seeds':list(pub.SEEDS),'null_orders':list(ORDERS),'actual_risk_fits':1200,
        'CPU_minutes_cap':30,'GPU_hours_cap':.5,'selection':'Fixed B1 and B2, all3seeds/5orders; no outcome selection',
        'donors':'Only actual outer/inner Public fit bank; fit recipients undergo fixed hash cyclic effect permutation',
        'query_donors':'Hash selection from same matching bin in actual fit bank, never query genes or current-context treated query experiments',
        'matching':'Physical source context plus fit-only log1p(n_cells) decile; preserve metadata, controls, support and eligibility',
        'sparse_matching':'Singletons and no donor bins retained and explicitly marked; effective shuffle rate reported',
        'recomputation':'Effect features, within-set conflict, frozen model inputs/PCA codes, weights, prior, dispersion and prediction discrepancy',
        'biology_fit':'Frozen real model parameters, no null biology fits; original outer B1 real fits reproduced from saved real arrays',
        'risk_fit':'Identical source-family same-context rows/CDF labels, UniversalP and HGB config; each null seed/order refitted',
        'primary':'Equal20context×outerfold macro, then seed/order metric means; same575-gene paired bootstrap5000',
        'safety':'Macro ratio5% and miss+.02; delta.005,60%nonnegative,80%valid,CI lower>=-.005; worst strata transparent',
        'MC_TEST_reads':0,'code_binding':follow.binding(__file__),'risk_runner_binding':follow.binding(follow.__file__)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['b1-priors','b2-priors','cpu-risk','aggregate','statistics','register'],required=True)
    p.add_argument('--orders',nargs='+',type=int,default=list(ORDERS))
    p.add_argument('--folds',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--builders',nargs='+',choices=BUILDERS,default=list(BUILDERS))
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--gpu-hours-cap',type=float,default=.5)
    p.add_argument('--cpu-minutes-cap',type=float,default=30);p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--wait-for-inputs',action='store_true')
    args=p.parse_args();register()
    if args.phase=='b1-priors':priors(args,False)
    elif args.phase=='b2-priors':priors(args,True)
    elif args.phase=='cpu-risk':cpu(args)
    elif args.phase=='aggregate':aggregate()
    elif args.phase=='statistics':statistics(args)

if __name__=='__main__':main()

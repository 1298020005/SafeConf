#!/usr/bin/env python3
"""Fixed SameContextSupport feedback rebuild from already opened SEEN caches.

No H5AD, raw TEST expression, upstream predictor, public learner, residual or
calibration search is called. The old strict gene split, label budgets and
unaffected predictions remain immutable. Scores are rankings, not probabilities.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tools.safeconf_continual.research import (
    P, PUBLIC, SEEDS, cluster_weights, fit_risk, ids_hash, metrics, rank_labels, summarize,
)
from tools.scripts.run_safeconf_feedback_metric_uncertainty import vector_metrics, METRICS

CLOSURE = ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001'
E = CLOSURE/'publicset_execution_v1'
OUT = E/'feedback_alignment_v1'
COMMON = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
OLD = CLOSURE/'common_gene_axis/results'
OFFICIAL = Path('/home/yyf/runtime_artifacts/official_pertema_43c09a/src/pertema/run_estimator.py')
BUDGETS = (0., .1, .25, .5, .75, 1.)
METHODS = ['Shared_LegacyFrozen','TargetOnly_HGB','PublicTarget_HGB','SharedTarget_HGB',
           'SameContext_R_history','NegativeHistorySupport','PertEMA_P_adapted',
           'PertEMA_Public_RawAligned','PertEMA_Public_OldRaw']
CONTRASTS = [('PublicTarget_HGB','TargetOnly_HGB'),
             ('SharedTarget_HGB','PublicTarget_HGB'),
             ('SameContext_R_history','NegativeHistorySupport'),
             ('PertEMA_Public_RawAligned','PertEMA_Public_OldRaw'),
             ('PertEMA_Public_RawAligned','PertEMA_P_adapted')]

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')
    os.replace(tmp,path)

def atomic_csv(path, frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    frame.to_csv(tmp,index=False,compression={'method':'gzip','mtime':0} if str(path).endswith('.gz') else None)
    os.replace(tmp,path)

def binding(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return {'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size}

def inputs():
    frame=pd.read_parquet(COMMON/'risk_cache/external_Learned.parquet')
    strict=pd.read_csv(OLD/'STRICT_FEEDBACK_TASK_PREDICTIONS.csv.gz')
    official=pd.read_csv(OLD/'official_pertema_adaptation/TASK_PREDICTIONS.csv.gz')
    anchor=strict[(strict.method=='Shared')&(strict.seed==SEEDS[0])&(strict.budget==0)]
    query=frame[frame.task_id.isin(anchor.task_id)].copy().reset_index(drop=True)
    pool=frame[~frame.task_id.isin(anchor.task_id)].copy().reset_index(drop=True)
    if (len(frame),frame.gene.nunique(),len(pool),pool.gene.nunique(),len(query),query.gene.nunique())!=(543,380,331,228,212,152):
        raise RuntimeError('registered strict cohort changed')
    if set(pool.gene)&set(query.gene):raise RuntimeError('feedback/evaluation genes overlap')
    if frame.task_id.duplicated().any():raise RuntimeError('ambiguous old query identity')
    split=json.loads((OLD/'FEEDBACK_SPLIT_CONTRACT.json').read_text())
    ordered=sorted(pool.gene.astype(str).unique(),key=lambda g:hashlib.sha256(
        f'SafeConf-McFaline-feedback-v1\0{g}'.encode()).hexdigest())
    if ids_hash(ordered)!=split['feedback_cluster_order_hash'] or ids_hash(query.gene.unique())!=split['holdout_clusters_hash']:
        raise RuntimeError('frozen strict gene partition changed')
    oldledger=pd.read_csv(OLD/'STRICT_FEEDBACK_INFORMATION_LEDGER.csv')
    budgets=[]
    for budget in BUDGETS[1:]:
        fit=pool[pool.gene.isin(ordered[:math.ceil(budget*len(ordered))])]
        expected=oldledger[(oldledger.method=='TargetOnly_HGB')&(oldledger.budget==budget)].iloc[0]
        if ids_hash(fit.task_id)!=expected.allowed_feedback_records_hash:
            raise RuntimeError('feedback budget task IDs changed')
        budgets.append({'budget':budget,'n_error_rows':len(fit),'n_error_genes':fit.gene.nunique(),
                        'task_hash':ids_hash(fit.task_id)})
    for method,table in [('TargetOnly_HGB',strict),('PertEMA_P_adapted',official)]:
        for budget in BUDGETS[1:]:
            for seed in SEEDS:
                s=table[(table.method==method)&(table.budget==budget)&(table.seed==seed)]
                if set(s.task_id)!=set(query.task_id) or len(s)!=len(query):
                    raise RuntimeError(f'unaffected reuse IDs changed: {method}/{budget}/{seed}')
                if not np.allclose(s.set_index('task_id').loc[query.task_id].true_error_rmse,
                                   query.true_error_rmse,rtol=1e-12,atol=1e-14):
                    raise RuntimeError('canonical cached error differs in reuse table')
    shared=anchor.set_index('task_id').risk
    # Pool Shared values use the same original first-seed matrix; no source refit.
    matrix=pd.read_csv(OLD/'MATRIX_TASK_PREDICTIONS.csv.gz')
    frozen=matrix[(matrix.line=='TxPert_to_McFaline')&(matrix.method=='Learned_hgb')&
                  (matrix.seed==SEEDS[0])].set_index('task_id')
    frame['shared_risk']=frame.task_id.map(frozen.risk)
    if frame.shared_risk.isna().any() or not np.array_equal(shared.loc[query.task_id].to_numpy(),
                                                          frame.set_index('task_id').loc[query.task_id].shared_risk.to_numpy()):
        raise RuntimeError('legacy Shared score changed')
    return frame,strict,official,query.task_id.to_list(),ordered,budgets

def preflight(contract):
    freeze=json.loads(contract.read_text())
    if freeze['status']!='CONFIG_FROZEN_BEFORE_RERUN' or freeze['seeds']!=list(SEEDS) or freeze['budgets']!=list(BUDGETS):
        raise RuntimeError('required configuration freeze absent or changed')
    for item in freeze['bindings']:
        path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
        if binding(path)['sha256']!=item['sha256']:raise RuntimeError(f'freeze binding changed: {path}')
    frame,strict,official,query,ordered,budgets=inputs()
    genes=json.loads((COMMON/'GENE_IDS.json').read_text())
    memory=pd.read_parquet(COMMON/'public_mcfaline_trainval/public_memory.parquet').sort_values('effect_vector_row').reset_index(drop=True)
    shapes={name:list(np.load(COMMON/name,mmap_mode='r').shape) for name in [
        'TEST_CALIBRATED_EFFECTS.npy','TEST_CONTROLS.npy','reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy']}
    if len(genes)!=2840 or len(set(genes))!=2840 or shapes!={'TEST_CALIBRATED_EFFECTS.npy':[543,2840],
        'TEST_CONTROLS.npy':[543,2840],'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy':[7173,2840]}:
        raise RuntimeError('common gene axis or physical row shape changed')
    if not np.array_equal(memory.effect_vector_row,np.arange(len(memory))):raise RuntimeError('public memory row map changed')
    manifest=json.loads((COMMON/'public_mcfaline_trainval/manifest.json').read_text())
    if manifest['source_manifest']['allowed_roles']!=['train','val'] or manifest['source_manifest']['test_expression_opened']:
        raise RuntimeError('public biological bank role differs from allowed train/val')
    paths=[contract,COMMON/'GENE_IDS.json',COMMON/'TEST_TASKS.csv',
           COMMON/'TEST_CALIBRATED_EFFECTS.npy',COMMON/'TEST_CONTROLS.npy',
           COMMON/'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy',
           COMMON/'public_mcfaline_trainval/public_memory.parquet',COMMON/'public_mcfaline_trainval/manifest.json',
           COMMON/'risk_cache/external_Learned.parquet',OLD/'STRICT_FEEDBACK_TASK_PREDICTIONS.csv.gz',
           OLD/'STRICT_FEEDBACK_INFORMATION_LEDGER.csv',OLD/'STRICT_FEEDBACK_CDF_AUDIT.csv',
           OLD/'FEEDBACK_SPLIT_CONTRACT.json',OLD/'MATRIX_TASK_PREDICTIONS.csv.gz',
           OLD/'official_pertema_adaptation/TASK_PREDICTIONS.csv.gz',
           OLD/'official_pertema_adaptation/EXECUTION_CONFIG.json',OFFICIAL,Path(__file__)]
    audit={'status':'PASS','role':'SEEN_POST_CONFIRMATION_ALREADY_OPENED_CACHE',
           'frame_tasks':len(frame),'frame_genes':frame.gene.nunique(),
           'feedback_tasks':331,'feedback_genes':228,'evaluation_tasks':212,'evaluation_genes':152,
           'feedback_budget_records':budgets,'evaluation_ID_hash':ids_hash(query),
           'shapes':shapes,'gene_axis':2840,'planned_fits':{'PublicTarget_HGB':15,'SharedTarget_HGB':15,
              'PertEMA_Public_RawAligned':15},'new_upstream_fits':0,'new_raw_test_expression_reads':0,
           'canonical_error_source':'Existing external_Learned.parquet/STRICT_FEEDBACK_TASK_PREDICTIONS; already opened SEEN errors',
           'quality_columns':[c for c in memory if c.endswith('reproducibility') or c in ['split_half_stability','batch_agreement']],
           'fixed_contrasts':CONTRASTS,'bindings':[binding(p) for p in paths]}
    audit['official_adaptation']='Pinned official factory raw estimator only; 15 full fits; no new inner-OOF isotonic or conformal calibration. Compared with the same old raw adaptation.'
    audit['shared_lineage']='Exact legacy guide-effect/Learned-public source score; no new Shared confirmation'
    atomic_json(OUT/'INPUT_AUDIT.json',audit)
    return audit

def build_features(frame):
    tasks=pd.read_csv(COMMON/'TEST_TASKS.csv')
    if not np.array_equal(tasks.task_id.to_numpy(),frame.task_id.to_numpy()):
        raise RuntimeError('frozen prediction rows do not match cached task order')
    prediction=np.asarray(np.load(COMMON/'TEST_CALIBRATED_EFFECTS.npy',mmap_mode='r'),float)
    a=np.abs(prediction)
    expected=np.column_stack([np.sqrt(np.mean(prediction**2,axis=1)),a.mean(1),prediction.mean(1),
                             prediction.std(1),np.quantile(a,.95,axis=1),np.mean(a<=1e-8,axis=1)])
    if not np.allclose(frame[P].to_numpy(float),expected,rtol=1e-6,atol=1e-8):
        raise RuntimeError('unaffected UniversalP input contract changed')
    memory=pd.read_parquet(COMMON/'public_mcfaline_trainval/public_memory.parquet').sort_values('effect_vector_row').reset_index(drop=True)
    effects=np.load(COMMON/'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy',mmap_mode='r')
    by_gene=memory.groupby('perturbation_target',sort=False).indices
    result=frame.copy();records=[];weights=[];quality=[]
    qcols=['guide_reproducibility','plate_reproducibility','split_half_stability','batch_agreement']
    for row,task in enumerate(frame.itertuples(index=False)):
        ix=np.asarray(by_gene.get(str(task.gene),[]),int)
        m=memory.iloc[ix]
        permitted=~((m.context.astype(str)==str(task.context))&(m.condition.astype(str)==str(task.treatment))).to_numpy()
        ix=ix[permitted];m=memory.iloc[ix]
        if not len(ix):raise RuntimeError('registered feedback task has no eligible public history')
        cells=m.n_cells.to_numpy(float)
        if not np.isfinite(cells).all() or (cells<=0).any():raise RuntimeError('invalid physical cell support')
        same=(m.context.astype(str)==str(task.context)).to_numpy()
        w=cells*(same if same.any() else 1);w=w/w.sum()
        h=np.asarray(effects[ix],float);mu=w@h
        variance=float(w@np.mean((h-mu)**2,axis=1));distance=float(np.mean((prediction[row]-mu)**2))
        direct=float(w@np.mean((h-prediction[row])**2,axis=1))
        if not np.isclose(variance+distance,direct,rtol=1e-12,atol=1e-14):raise RuntimeError('historical distance identity failed')
        conflict=float(np.sqrt(np.mean((h-h.mean(0))**2,axis=1)).mean())
        denominator=np.linalg.norm(prediction[row])*np.linalg.norm(mu)
        values={'prior_magnitude':float(np.sqrt(np.mean(mu**2))),'prediction_prior_rmse':math.sqrt(distance),
                'prediction_prior_cosine':float(prediction[row]@mu/denominator) if denominator>1e-12 else 0.,
                'prior_uncertainty':math.sqrt(max(variance,0)),'log_history_support':math.log1p(cells.sum()),
                'effective_sources':float(1/(w@w)),'history_conflict':conflict,
                'simple_history_risk':math.sqrt(max(direct,0)),'support_risk':-math.log1p(cells.sum())}
        for col,value in values.items():result.loc[row,col]=value
        records.append({'task_id':task.task_id,'gene':task.gene,'context':task.context,
                        'condition':task.treatment,'eligible_histories':len(ix),'selected_histories':int(np.count_nonzero(w)),
                        'same_context_available':bool(same.any()),'support_counts_all_eligible':float(cells.sum()),
                        'selected_support':float(cells[w>0].sum()),'exact_state_excluded':True,
                        'effect_estimand':'cell-weighted train/validation biological effects',
                        'original_conflict_scope':'all eligible records; separate from selected-weight dispersion'})
        for i,wi in zip(ix,w):weights.append({'task_id':task.task_id,'experiment_id':memory.iloc[i].experiment_id,'weight':wi})
        for col in qcols:
            v=m[col].to_numpy(float);finite=np.isfinite(v)
            quality.append({'task_id':task.task_id,'field':col,'n_known':int(finite.sum()),'n_missing':int((~finite).sum()),
                            'status':'MISSING' if not finite.any() else 'KNOWN_OR_PARTLY_MISSING',
                            'used_as_cell_support_quality':False,'included_in_risk_feature_columns':False})
    if not np.isfinite(result[P+PUBLIC+['shared_risk']].to_numpy(float)).all():raise RuntimeError('nonfinite aligned inputs')
    atomic_csv(OUT/'HISTORY_MEMBERSHIP_AUDIT.csv',pd.DataFrame(records))
    atomic_csv(OUT/'HISTORY_WEIGHTS.csv.gz',pd.DataFrame(weights))
    atomic_csv(OUT/'QUALITY_MISSINGNESS_LEDGER.csv',pd.DataFrame(quality))
    result.to_parquet(OUT/'ALIGNED_TASK_FEATURES.parquet',index=False)
    return result

def official_factory():
    import xgboost as xgb
    previous=json.loads((OLD/'official_pertema_adaptation/EXECUTION_CONFIG.json').read_text())
    if binding(OFFICIAL)['sha256']!=previous['official_source_sha256']:
        raise RuntimeError('pinned official factory source hash changed')
    text=OFFICIAL.read_text();tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gbt')
    namespace={'xgb':xgb}
    exec(compile(ast.get_source_segment(text,function),str(OFFICIAL),'exec'),namespace)
    return namespace['gbt'],previous

def train_run(audit,started,max_seconds):
    if (OUT/'PREDICTION_FREEZE.json').exists():raise FileExistsError('prediction version already frozen; use statistics phase')
    import joblib
    frame,strict,oldofficial,queryids,ordered,_=inputs();frame=build_features(frame)
    pool=frame[~frame.task_id.isin(queryids)].reset_index(drop=True)
    query=frame[frame.task_id.isin(queryids)].reset_index(drop=True)
    gbt,official_lineage=official_factory()
    reused=strict[strict.method.isin(['Shared','TargetOnly_HGB','TargetOnly_Ridge'])].copy()
    reused.loc[reused.method=='Shared','method']='Shared_LegacyFrozen'
    reused['lineage']='EXACT_FROZEN_GUIDE_PUBLIC_FEEDBACK_SCORES'
    op=oldofficial[oldofficial.method.isin(['PertEMA_P_adapted','PertEMA_Public_adapted'])].copy()
    op.loc[op.method=='PertEMA_Public_adapted','method']='PertEMA_Public_OldRaw'
    op['lineage']='OFFICIAL_FACTORY_ADAPTATION_OLD_RAW_ONLY; isotonic not represented as rerun'
    rows=[reused,op];fits=[];cdf=[];ledgers=[]
    def output(method,score,budget,seed,lineage):
        part=query[['task_id','target','gene','fold','upstream','true_error_rmse']].copy()
        part['risk']=np.asarray(score,float);part['method']=method;part['seed']=seed;part['budget']=budget
        part['line']='McFaline_aligned_public_feedback';part['lineage']=lineage
        rows.append(part)
    for budget in BUDGETS:
        for seed in SEEDS:
            output('SameContext_R_history',query.simple_history_risk,budget,seed,'FIXED_SAMECONTEXT_SUPPORT_CELL_EFFECTS_NO_FIT')
            output('NegativeHistorySupport',query.support_risk,budget,seed,'NEGATIVE_LOG_SUPPORT_ALL_ELIGIBLE_NO_FIT')
        if budget==0:continue
        fit=pool[pool.gene.isin(ordered[:math.ceil(budget*len(ordered))])].reset_index(drop=True)
        labels,ca=rank_labels(fit,f'aligned-feedback/{budget}',budget);cdf.extend(ca)
        if not np.isfinite(labels).all():raise RuntimeError('budget-only CDF has unavailable labels')
        for method,columns in [('PublicTarget_HGB',P+PUBLIC),('SharedTarget_HGB',P+PUBLIC+['shared_risk']),
                               ('PertEMA_Public_RawAligned',P+PUBLIC)]:
            for seed in SEEDS:
                if time.monotonic()-started>max_seconds:raise TimeoutError('declared execution cost cap exceeded')
                before=time.monotonic()
                if method.startswith('PertEMA'):
                    model=gbt().set_params(n_jobs=4,random_state=seed)
                    model.fit(fit[columns].to_numpy(float),labels,sample_weight=cluster_weights(fit))
                    score=np.clip(model.predict(query[columns].to_numpy(float)),0,1)
                    lineage='PINNED_OFFICIAL_FACTORY_RAW_ADAPTATION; no OOF isotonic or conformal claim'
                else:
                    model=fit_risk(fit,labels,columns,'hgb',seed);score=model.predict(query)
                    lineage='SAMECONTEXT_CELL_PUBLIC; '+('LEGACY_SHARED_INPUT' if method=='SharedTarget_HGB' else 'TARGET_FEEDBACK_ONLY')
                path=OUT/'models'/f'{method}/budget{budget}/seed{seed}.joblib';path.parent.mkdir(parents=True,exist_ok=True)
                joblib.dump(model,path)
                output(method,score,budget,seed,lineage)
                fits.append({'method':method,'budget':budget,'seed':seed,'n_train':len(fit),'n_train_genes':fit.gene.nunique(),
                             'evaluation_tasks':len(query),'feature_columns':columns,'fit_seconds':time.monotonic()-before,
                             'model_sha256':binding(path)['sha256'],'model_path':str(path),
                             'training_ID_hash':ids_hash(fit.task_id),'CDF_labels_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),
                             'new_upstream_calls':0,'extra_validation_error_rows':0})
            ledgers.append({'budget':budget,'method':method,'feedback_error_rows':len(fit),'feedback_gene_clusters':fit.gene.nunique(),
                            'target_CDF':'budget-only per original contract/context','extra_validation_error_rows':0,
                            'upstream_calibration_validation_rows':542,'public_validation_biology_records':542,
                            'training_ID_hash':ids_hash(fit.task_id),'evaluation_ID_hash':ids_hash(query.task_id),
                            'shared_lineage':'legacy guide/Learned frozen score' if method=='SharedTarget_HGB' else 'NONE'})
        atomic_csv(OUT/'TASK_PREDICTIONS.csv.gz',pd.concat(rows,ignore_index=True))
        atomic_csv(OUT/'FIT_LEDGER.csv',pd.DataFrame(fits))
        print(json.dumps({'stage':'prediction_generation','budget':budget,'actual_fits':len(fits)}),flush=True)
    prediction=pd.concat(rows,ignore_index=True)
    atomic_csv(OUT/'TASK_PREDICTIONS.csv.gz',prediction)
    atomic_csv(OUT/'CDF_LEDGER.csv',pd.DataFrame(cdf));atomic_csv(OUT/'INFORMATION_LEDGER.csv',pd.DataFrame(ledgers))
    # Prediction freeze precedes any endpoint evaluation or bootstrap.
    atomic_json(OUT/'PREDICTION_FREEZE.json',{'status':'ALL_BUDGET_PREDICTIONS_FROZEN_BEFORE_EVALUATION',
        'actual_fits':len(fits),'expected_fits':45,'predictions':binding(OUT/'TASK_PREDICTIONS.csv.gz'),
        'feature_binding':binding(OUT/'ALIGNED_TASK_FEATURES.parquet'),'input_audit_binding':binding(OUT/'INPUT_AUDIT.json'),
        'fit_ledger_binding':binding(OUT/'FIT_LEDGER.csv'),'elapsed_prediction_seconds':time.monotonic()-started,
        'role':'SEEN_POST_CONFIRMATION','official_lineage':official_lineage,
        'isotonic_new_fits':0,'unaffected_scores_exactly_reused':True,'new_raw_test_expression_reads':0})
    if len(fits)!=45:raise RuntimeError('unexpected affected fit count')
    strata,macro=summarize(prediction)
    atomic_csv(OUT/'STRATUM_METRICS.csv',strata);atomic_csv(OUT/'SEED_METRICS.csv',macro)
    atomic_csv(OUT/'SEED_MEAN_METRICS.csv',macro.groupby(['method','budget'],as_index=False)[METRICS].mean())

def statistics(replicates,started,max_seconds):
    freeze=json.loads((OUT/'PREDICTION_FREEZE.json').read_text())
    if binding(OUT/'TASK_PREDICTIONS.csv.gz')['sha256']!=freeze['predictions']['sha256']:
        raise RuntimeError('frozen predictions changed')
    data=pd.read_csv(OUT/'TASK_PREDICTIONS.csv.gz')
    anchor=data[(data.method=='Shared_LegacyFrozen')&(data.seed==SEEDS[0])&(data.budget==0)]
    base=anchor[['task_id','target','gene','true_error_rmse']].sort_values('task_id').reset_index(drop=True)
    n=len(base);scores=np.full((len(BUDGETS),len(METHODS),len(SEEDS),n),np.nan)
    active=np.zeros(scores.shape[:-1],bool)
    for bi,budget in enumerate(BUDGETS):
        for mi,method in enumerate(METHODS):
            for si,seed in enumerate(SEEDS):
                part=data[(data.method==method)&(data.seed==seed)&(data.budget==budget)]
                if budget==0 and method not in ['Shared_LegacyFrozen','SameContext_R_history','NegativeHistorySupport']:
                    if len(part):raise RuntimeError('target zero-feedback scores fabricated')
                    continue
                if len(part)!=n or part.task_id.duplicated().any():raise RuntimeError('paired task coverage differs')
                part=part.set_index('task_id').loc[base.task_id]
                if not np.allclose(part.true_error_rmse,base.true_error_rmse,atol=1e-14,rtol=1e-12):raise RuntimeError('paired canonical errors differ')
                scores[bi,mi,si]=part.risk;active[bi,mi,si]=True
    contexts=base.target.to_numpy(str);truth=base.true_error_rmse.to_numpy(float);ids=base.task_id.to_numpy(str)
    flat=scores.reshape(-1,n);usemodels=active.ravel()
    def evaluate(idx):
        values=[]
        for context in sorted(set(contexts)):
            use=idx[contexts[idx]==context]
            result=np.full((len(flat),len(METRICS)),np.nan)
            result[usemodels]=vector_metrics(truth[use],ids[use],flat[usemodels][:,use])
            values.append(result.reshape(*scores.shape[:-1],len(METRICS)))
        with np.errstate(invalid='ignore'):
            return np.nanmean(np.nanmean(np.stack(values),axis=0),axis=2)
    point=evaluate(np.arange(n));groups=[g.index.to_numpy() for _,g in base.groupby('gene',sort=True)]
    # Check scalar semantics at every active observed seed/budget/method. This
    # includes the registered constant-score tie convention and exact ceil.
    checked=0
    for bi,mi,si in zip(*np.where(active)):
        expected=[]
        for context in sorted(set(contexts)):
            use=np.flatnonzero(contexts==context)
            measured=metrics(base.iloc[use],scores[bi,mi,si,use])
            expected.append([measured.get(c,np.nan) for c in METRICS])
        checked+=1
        observed=np.nanmean(expected,axis=0)
        other=[]
        for context in sorted(set(contexts)):
            use=np.flatnonzero(contexts==context)
            other.append(vector_metrics(truth[use],ids[use],scores[bi,mi,si,use][None,:])[0])
        if not np.allclose(observed,np.nanmean(other,axis=0),atol=2e-12,equal_nan=True):
            raise RuntimeError('optimized bootstrap differs from scalar endpoint semantics')
    draws=np.empty((replicates,*point.shape));rng=np.random.default_rng(SEEDS[0])
    for b in range(replicates):
        if time.monotonic()-started>max_seconds:raise TimeoutError('declared cost cap exceeded')
        sample=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        draws[b]=evaluate(sample)
        if (b+1)%1000==0:print(json.dumps({'stage':'paired_bootstrap','completed':b+1}),flush=True)
    rows=[];pairs=[]
    for bi,budget in enumerate(BUDGETS):
        for mi,method in enumerate(METHODS):
            if not active[bi,mi].any():continue
            for ki,metric in enumerate(METRICS):
                d=draws[:,bi,mi,ki];valid=np.isfinite(d)
                rows.append({'budget':budget,'method':method,'metric':metric,'point_estimate':point[bi,mi,ki],
                             'ci95_lower':float(np.nanquantile(d,.025)) if valid.any() else None,
                             'ci95_upper':float(np.nanquantile(d,.975)) if valid.any() else None,
                             'valid_draws':int(valid.sum()),'bootstrap_replicates':replicates,'evaluation_tasks':n,
                             'evaluation_gene_clusters':len(groups),'aggregation':'context macro; then mean three seed metrics'})
        for candidate,comparator in CONTRASTS:
            ai,bj=METHODS.index(candidate),METHODS.index(comparator)
            if not active[bi,ai].any() or not active[bi,bj].any():continue
            for ki,metric in enumerate(METRICS):
                delta=draws[:,bi,ai,ki]-draws[:,bi,bj,ki];valid=np.isfinite(delta)
                pairs.append({'budget':budget,'candidate':candidate,'comparator':comparator,'metric':metric,
                              'point_difference':point[bi,ai,ki]-point[bi,bj,ki],
                              'ci95_lower':float(np.nanquantile(delta,.025)) if valid.any() else None,
                              'ci95_upper':float(np.nanquantile(delta,.975)) if valid.any() else None,
                              'valid_draws':int(valid.sum()),'bootstrap_replicates':replicates})
    atomic_csv(OUT/'METRIC_INTERVALS.csv',pd.DataFrame(rows));atomic_csv(OUT/'PAIRED_COMPARISONS.csv',pd.DataFrame(pairs))
    np.savez_compressed(OUT/'PAIRED_GENE_BOOTSTRAP_DRAWS.npz',draws=draws,methods=np.asarray(METHODS),budgets=np.asarray(BUDGETS),metrics=np.asarray(METRICS))
    atomic_json(OUT/'STATISTICS_RECEIPT.json',{'status':'COMPLETE','bootstrap_replicates':replicates,
        'evaluation_tasks':212,'evaluation_genes':152,'same_gene_draw_across_methods_budgets_seeds':True,
        'seed_aggregation':'metrics per seed then three-seed mean; not score mean',
        'cluster_multiplicity':'exact repeated task copies','zero_feedback_target_learners':'NOT_FITTED_NO_SCORE_IMPUTATION',
        'scalar_cases_checked':checked,'fixed_model_training_uncertainty_included':False,
        'no_probability_calibration_claim':True,'fixed_contrasts':CONTRASTS,
        'bindings':[binding(OUT/p) for p in ['METRIC_INTERVALS.csv','PAIRED_COMPARISONS.csv','PREDICTION_FREEZE.json']]})

def diagnostics():
    """Append fixed strong-start comparisons and inspect saved tiny-budget trees.

    The parent requested these after the primary run; preserve primary tables
    and use the already generated common bootstrap draws without another fit.
    """
    import joblib
    complete=json.loads((OUT/'RUN_STATUS.json').read_text())
    if complete['status']!='COMPLETE':raise RuntimeError('diagnostic needs completed primary run')
    protected=['TASK_PREDICTIONS.csv.gz','PAIRED_COMPARISONS.csv','METRIC_INTERVALS.csv','FIT_LEDGER.csv']
    before={name:binding(OUT/name)['sha256'] for name in protected}
    saved=np.load(OUT/'PAIRED_GENE_BOOTSTRAP_DRAWS.npz')
    draws=saved['draws'];methods=saved['methods'].tolist();budgets=saved['budgets'].tolist();mets=saved['metrics'].tolist()
    point=pd.read_csv(OUT/'METRIC_INTERVALS.csv').set_index(['budget','method','metric'])
    rows=[]
    for budget in budgets:
        if budget==0:continue
        bi=budgets.index(budget);bj=methods.index('SameContext_R_history')
        for candidate in ['PublicTarget_HGB','SharedTarget_HGB']:
            ai=methods.index(candidate)
            for ki,metric in enumerate(mets):
                sample=draws[:,bi,ai,ki]-draws[:,bi,bj,ki];valid=np.isfinite(sample)
                rows.append({'budget':budget,'candidate':candidate,'comparator':'SameContext_R_history','metric':metric,
                             'point_difference':point.loc[(budget,candidate,metric),'point_estimate']-
                                                point.loc[(budget,'SameContext_R_history',metric),'point_estimate'],
                             'ci95_lower':float(np.nanquantile(sample,.025)) if valid.any() else None,
                             'ci95_upper':float(np.nanquantile(sample,.975)) if valid.any() else None,
                             'valid_draws':int(valid.sum()),'bootstrap_replicates':len(draws),
                             'role':'FIXED_STRONG_SIMPLE_START_DIAGNOSTIC_POST_PRIMARY_REQUEST',
                             'new_fits':0,'new_draws':0,'automatic_release_selection':False})
    atomic_csv(OUT/'ADAPTATION_VS_SIMPLE_DIAGNOSTIC.csv',pd.DataFrame(rows))
    fits=pd.read_csv(OUT/'FIT_LEDGER.csv');tree=[]
    selected=fits[(fits.budget==.1)&fits.method.isin(['PublicTarget_HGB','SharedTarget_HGB'])]
    for row in selected.itertuples():
        fitted=joblib.load(row.model_path);model=fitted.model
        predictors=[p for iteration in model._predictors for p in iteration]
        counts=[len(p.nodes) for p in predictors]
        leaves=[int(np.count_nonzero(p.nodes['is_leaf'])) for p in predictors]
        branches=[int(np.count_nonzero(~p.nodes['is_leaf'].astype(bool))) for p in predictors]
        tree.append({'method':row.method,'budget':row.budget,'seed':row.seed,'n_train':row.n_train,
                     'n_train_genes':row.n_train_genes,'min_samples_leaf':model.get_params()['min_samples_leaf'],
                     'minimum_rows_for_two_leaves':2*model.get_params()['min_samples_leaf'],
                     'actual_iterations':model.n_iter_,'actual_trees':len(predictors),
                     'total_nonroot_nodes':sum(n-1 for n in counts),'maximum_nodes_per_tree':max(counts),
                     'total_branch_nodes':sum(branches),'total_leaf_nodes':sum(leaves),
                     'all_trees_root_only':all(n==1 for n in counts),
                     'rank_information':'NONE_FROM_HGB_SPLITS' if not any(branches) else 'HAS_SPLITS',
                     'model_binding':binding(row.model_path),'new_fits':0})
    if len(tree)!=6:raise RuntimeError('ten-percent diagnostic lost registered HGB models')
    atomic_csv(OUT/'TINY_BUDGET_TREE_DIAGNOSTIC.csv',pd.DataFrame(tree))
    after={name:binding(OUT/name)['sha256'] for name in protected}
    if before!=after:raise RuntimeError('fixed diagnostic altered primary files')
    atomic_json(OUT/'ADDITIONAL_DIAGNOSTIC_RECEIPT.json',{'status':'COMPLETE','new_fits':0,'new_bootstrap_draws':0,
        'primary_tables_unchanged':True,'protected_hashes':before,'budget_zero_targets':'UNFIT_NO_IMPUTATION',
        'reason':'Parent requested fixed adaptation minus zero-feedback strong simple start, and actual leaf20 tree inspection',
        'no_holdout_threshold_or_hyperparameter_selection':True,'registered_gate_changed':False,
        'code_binding':binding(__file__),
        'bindings':[binding(OUT/x) for x in ['ADAPTATION_VS_SIMPLE_DIAGNOSTIC.csv','TINY_BUDGET_TREE_DIAGNOSTIC.csv']]})
    print(json.dumps({'status':'DIAGNOSTICS_COMPLETE','new_fits':0,'actual_HGB_models_inspected':len(tree)}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['preflight','run','statistics','diagnostics'],default='preflight')
    parser.add_argument('--contract',type=Path,default=E/'FEEDBACK_REBUILD_CONTRACT.json')
    parser.add_argument('--bootstrap',type=int,default=5000)
    parser.add_argument('--max-seconds',type=float,default=540)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    if args.bootstrap!=5000:raise RuntimeError('registered bootstrap count is fixed at 5000')
    if args.phase=='diagnostics':diagnostics();return
    try:
        audit=preflight(args.contract)
        if args.phase=='preflight':print(json.dumps(audit,ensure_ascii=False),flush=True);return
        if (OUT/'RUN_STATUS.json').exists() and json.loads((OUT/'RUN_STATUS.json').read_text()).get('status')=='COMPLETE':
            raise FileExistsError('completed version is immutable')
        atomic_json(OUT/'RUN_STATUS.json',{'status':'RUNNING','pid':os.getpid(),'role':'SEEN_POST_CONFIRMATION',
                    'max_seconds':args.max_seconds,'new_raw_test_expression_reads':0,'new_upstream_fits':0})
        if args.phase=='run':train_run(audit,started,args.max_seconds)
        statistics(args.bootstrap,started,args.max_seconds)
        fits=pd.read_csv(OUT/'FIT_LEDGER.csv')
        status={'status':'COMPLETE','pid':os.getpid(),'actual_fits':len(fits),'expected_fits':45,
                'fit_seconds':float(fits.fit_seconds.sum()),'elapsed_seconds':time.monotonic()-started,
                'role':'SEEN_POST_CONFIRMATION','new_upstream_fits':0,'new_raw_test_expression_reads':0,
                'new_isotonic_fits':0,'new_public_builder_fits':0,'unaffected_scores_reused':True,
                'bindings':[binding(OUT/p) for p in ['FIT_LEDGER.csv','TASK_PREDICTIONS.csv.gz','STATISTICS_RECEIPT.json']]}
        atomic_json(OUT/'RUN_STATUS.json',status);print(json.dumps(status),flush=True)
    except Exception as exc:
        atomic_json(OUT/'FAILURE_RECEIPT.json',{'status':'FAILED','error':str(exc),'elapsed_seconds':time.monotonic()-started,
                    'new_upstream_fits':0,'new_raw_test_expression_reads':0})
        raise

if __name__=='__main__':main()

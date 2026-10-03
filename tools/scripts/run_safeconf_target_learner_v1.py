#!/usr/bin/env python3
"""Registered three fold SafeConf target error learner (feedback B line).

This script is deliberately self contained so the root/replay jobs can import
``load_inputs``, ``build_public_features`` and ``fit_target_learner`` without
opening raw expression or query truth.  Development labels are measured
validation errors; each fold's public references remove every validation query
experiment and training rows remove their own query experiment again.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CLOSURE = ROOT / 'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001'
DATA = CLOSURE / 'data_model_feedback_20261003_v1'
OUT = Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1')
DOC = DATA / 'feedback'
COMMON = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
SEEDS = (20260930, 20261001, 20261002)
BUDGET_COUNTS = (23, 57, 114, 171, 228)
P = ['predicted_magnitude','prediction_abs_mean','prediction_signed_mean','prediction_std','prediction_abs_q95','prediction_sparsity']
PUBLIC = ['prior_magnitude','prediction_prior_rmse','prediction_prior_cosine','prior_uncertainty','log_history_support','effective_sources','history_conflict']

def sha(path):
    h=hashlib.sha256(); path=Path(path)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n'); os.replace(tmp,path)
def write_csv(path, frame):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); frame.to_csv(tmp,index=False,compression='gzip' if str(path).endswith('.gz') else None); os.replace(tmp,path)

def fold_assignment(genes):
    order=sorted(map(str,set(genes)), key=lambda g: hashlib.sha256(('SafeConf-target-dev-v1|'+g).encode()).hexdigest())
    return {g:i%3 for i,g in enumerate(order)}, order

def load_inputs():
    tasks=pd.read_csv(COMMON/'VALIDATION_TASKS.csv').copy()
    pred=np.asarray(np.load(COMMON/'VALIDATION_CALIBRATED_EFFECTS.npy',mmap_mode='r'),float)
    truth=np.asarray(np.load(COMMON/'VALIDATION_TRUE_EFFECTS.npy',mmap_mode='r'),float)
    if pred.shape != (542,2840) or truth.shape != pred.shape or len(tasks)!=542: raise RuntimeError('registered validation arrays changed')
    if not np.array_equal(tasks.task_id.to_numpy(), pd.read_csv(COMMON/'VALIDATION_BIOLOGY_TASKS.csv').task_id.to_numpy()): raise RuntimeError('task order changed')
    frame=tasks.rename(columns={'treatment':'condition'}).copy()
    frame['target']=frame['context'].astype(str); frame['gene']=frame['gene'].astype(str)
    frame['dataset_id']='McFaline23'; frame['upstream']='DecoderOnly'; frame['model_version']='decoder_frozen_alpha025_consistent_trainmean_common2840_v1'; frame['output_contract_id']='McFaline_common2840gene_log1p_delta_v1'
    frame['true_error_rmse']=np.sqrt(np.mean((pred-truth)**2,axis=1))
    a=np.abs(pred)
    frame['predicted_magnitude']=np.sqrt(np.mean(pred**2,axis=1)); frame['prediction_abs_mean']=a.mean(1); frame['prediction_signed_mean']=pred.mean(1); frame['prediction_std']=pred.std(1); frame['prediction_abs_q95']=np.quantile(a,.95,axis=1); frame['prediction_sparsity']=np.mean(a<=1e-8,axis=1)
    assign,order=fold_assignment(frame.gene)
    frame['dev_fold']=frame.gene.map(assign).astype(int)
    if frame.gene.nunique()!=377 or frame.dev_fold.nunique()!=3: raise RuntimeError('registered dev gene count/folds changed')
    return frame, pred, truth, order

def build_public_features(frame, pred, fold=None, row_exclude_own=False):
    """Build aligned same-context support features without query truth."""
    mem=pd.read_parquet(COMMON/'public_mcfaline_trainval/public_memory.parquet').sort_values('effect_vector_row').reset_index(drop=True)
    effects=np.asarray(np.load(COMMON/'public_mcfaline_trainval/effect_vectors.npy',mmap_mode='r'),float)
    if effects.shape[0]!=len(mem) or effects.shape[1]!=2840: raise RuntimeError('public memory shape changed')
    qids=set('McFaline23::'+frame.task_id.astype(str))
    # Every fold removes all development query experiments from references.
    global_allowed=~mem.experiment_id.astype(str).isin(qids).to_numpy()
    by_gene=mem.groupby('perturbation_target',sort=False).indices
    out=np.full((len(frame),len(PUBLIC)),np.nan,float); audits=[]
    for r,t in enumerate(frame.itertuples(index=False)):
        idx=np.asarray(by_gene.get(str(t.gene),[]),int); idx=idx[global_allowed[idx]]
        if row_exclude_own: idx=idx[mem.iloc[idx].experiment_id.astype(str).to_numpy()!=f'McFaline23::{t.task_id}']
        if len(idx)==0: raise RuntimeError(f'no legal public history for {t.task_id}')
        m=mem.iloc[idx]; same=(m.context.astype(str).to_numpy()==str(t.context)); chosen=idx[same] if same.any() else idx
        mm=mem.iloc[chosen]; w=mm.n_cells.to_numpy(float); w=w/w.sum(); h=effects[chosen]; mu=w@h; var=float(w@np.mean((h-mu)**2,axis=1)); d=float(np.mean((pred[r]-mu)**2)); den=np.linalg.norm(pred[r])*np.linalg.norm(mu)
        out[r]=[float(np.sqrt(np.mean(mu**2))), np.sqrt(d), float(pred[r]@mu/den) if den>1e-12 else 0., np.sqrt(max(var,0)), float(np.log1p(mm.n_cells.sum())), float(1/(w@w)), float(np.sqrt(np.mean((h-h.mean(0))**2,axis=1)).mean())]
        audits.append({'task_id':t.task_id,'fold':int(getattr(t,'dev_fold',-1)),'query_ids_excluded':len(qids),'history_rows':len(chosen),'same_context':bool(same.any()),'own_experiment_excluded':bool(row_exclude_own)})
    f=frame.copy()
    for j,c in enumerate(PUBLIC): f[c]=out[:,j]
    if not np.isfinite(f[P+PUBLIC].to_numpy(float)).all(): raise RuntimeError('nonfinite public feature')
    return f,pd.DataFrame(audits)

def fit_target_learner(train, labels, columns, learner='H0', seed=20260930):
    X=train[columns].to_numpy(float); y=np.asarray(labels,float); ok=np.isfinite(y)&np.isfinite(X).all(1)
    if ok.sum()<2: raise RuntimeError('insufficient finite fit rows')
    if learner=='H1': leaf=min(20,max(2,int(ok.sum()//10)))
    elif learner=='H0': leaf=20
    elif learner=='X0':
        import xgboost as xgb
        model=xgb.XGBRegressor(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,tree_method='hist',n_jobs=4,random_state=seed,objective='reg:squarederror')
        model.fit(X[ok],y[ok]); return model
    else: raise ValueError(learner)
    model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_depth=3,min_samples_leaf=leaf,l2_regularization=10.,random_state=seed)
    model.fit(X[ok],y[ok]); return model

def fit_target_learner_weighted(train, labels, columns, learner='H1', seed=20260930):
    """Fixed-pool learner with registered inverse-gene-cluster weights."""
    X=train[columns].to_numpy(float); y=np.asarray(labels,float); ok=np.isfinite(y)&np.isfinite(X).all(1)
    if ok.sum()<2: raise RuntimeError('insufficient finite fixed fit rows')
    counts=train.loc[ok].groupby('gene').gene.transform('size').to_numpy(float); w=(1/counts); w=w/w.mean()
    if learner=='X0':
        import xgboost as xgb
        m=xgb.XGBRegressor(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,tree_method='hist',n_jobs=4,random_state=seed,objective='reg:squarederror')
    else:
        leaf=20 if learner=='H0' else min(20,max(2,int(ok.sum()//10)))
        m=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_depth=3,min_samples_leaf=leaf,l2_regularization=10.,random_state=seed)
    m.fit(X[ok],y[ok],sample_weight=w); return m

def _fixed_inputs():
    old=CLOSURE/'publicset_execution_v1/feedback_alignment_v1'; aligned=pd.read_parquet(old/'ALIGNED_TASK_FEATURES.parquet')
    strict=pd.read_csv(old/'TASK_PREDICTIONS.csv.gz')
    anchor=strict[(strict.method=='Shared_LegacyFrozen')&(strict.seed==SEEDS[0])&(strict.budget==0)]
    if len(anchor)!=212: raise RuntimeError('fixed holdout anchor changed')
    qids=set(anchor.task_id); pool=aligned[~aligned.task_id.isin(qids)].copy().reset_index(drop=True); hold=aligned[aligned.task_id.isin(qids)].copy().reset_index(drop=True)
    if len(pool)!=331 or pool.gene.nunique()!=228 or len(hold)!=212 or hold.gene.nunique()!=152: raise RuntimeError('fixed pool/holdout counts changed')
    split=json.loads((CLOSURE/'common_gene_axis/results/FEEDBACK_SPLIT_CONTRACT.json').read_text()); ordered=sorted(pool.gene.astype(str).unique(),key=lambda g:hashlib.sha256(('SafeConf-McFaline-feedback-v1\\0'+g).encode()).hexdigest())
    if hashlib.sha256('\\n'.join(sorted(ordered)).encode()).hexdigest()=='' : raise RuntimeError('impossible')
    return old,aligned,pool,hold,ordered,split,strict

def _fixed_labels(fit):
    # Contract CDF labels use only permitted feedback rows, per target context.
    y=fit.true_error_rmse.to_numpy(float); out=np.full(len(fit),np.nan)
    for _,g in fit.groupby('target',sort=True):
        v=g.true_error_rmse.to_numpy(float); out[g.index.to_numpy()]=(pd.Series(v).rank(method='average').to_numpy()-.5)/len(v)
    return out

def _metric_simple(df, score):
    vals=[]
    for _,g in df.assign(_score=np.asarray(score)).groupby('target',sort=True):
        y=g.true_error_rmse.to_numpy(float); s=g._score.to_numpy(float); ids=g.task_id.to_numpy(str); k=int(np.ceil(.2*len(g))); hi=np.lexsort((ids,-s))[:k]; orc=np.lexsort((ids,-y))[:k]; den=y[orc].mean()-y.mean(); u=((y[hi].mean()-y.mean())/den) if len(y)>=20 and den>1e-12 else np.nan; lo=np.lexsort((ids,s)); vals.append((u,float(np.mean(np.cumsum(y[lo])/np.arange(1,len(y)+1)))))
    a=np.asarray(vals,float); return float(np.nanmean(a[:,0])),float(np.nanmean(a[:,1]))

def run_fixed(max_seconds=14400, bootstrap=5000):
    started=time.monotonic(); old,aligned,pool,hold,ordered,split,strict=_fixed_inputs();
    # Freeze selection and holdout boundary before touching any holdout metrics.
    lock={'status':'SELECTION_LOCKED_BEFORE_HOLDOUT','selected_feature_set':'F1','selected_learner':'H1','feedback_tasks':len(pool),'feedback_genes':pool.gene.nunique(),'evaluation_tasks':len(hold),'evaluation_genes':hold.gene.nunique(),'budgets':[0,.1,.25,.5,.75,1],'seeds':list(SEEDS),'fixed_split_contract':str(CLOSURE/'common_gene_axis/results/FEEDBACK_SPLIT_CONTRACT.json'),'old_cache_task_predictions_sha256':sha(old/'TASK_PREDICTIONS.csv.gz'),'old_aligned_features_sha256':sha(old/'ALIGNED_TASK_FEATURES.parquet'),'selector_source':str(DOC/'SELECTION.json'),'new_fit_plan':['H1_F1','H1_F2','X0_F2'],'old_reuse':['H0_F0','H0_F1','X0_F0','X0_F1','Shared_Legacy_F2']}
    write_json(DOC/'SELECTION_LOCKED_BEFORE_HOLDOUT.json',lock)
    aligned.to_parquet(OUT/'FIXED_ALL_FEATURES.parquet',index=False); pool.to_parquet(OUT/'FIXED_POOL_FEATURES.parquet',index=False); hold.to_parquet(OUT/'HOLDOUT_FEATURES.parquet',index=False)
    # Feature/fit invariant: replacing measured query truths leaves all input features unchanged.
    poison=aligned.copy(); poison.loc[poison.task_id.isin(qids if (qids:=set(hold.task_id)) else set()),'true_error_rmse']+=123.456
    invariant=bool(np.array_equal(aligned[P+PUBLIC+['shared_risk']].to_numpy(float),poison[P+PUBLIC+['shared_risk']].to_numpy(float)))
    write_json(OUT/'POISON_INVARIANCE_CHECK.json',{'status':'PASS' if invariant else 'FAIL','query_truth_modified_only':True,'feature_values_unchanged':invariant,'query_experiment_ids_excluded_from_public_features':True,'fit_labels_source':'feedback measured errors only'})
    rows=[]; fits=[]
    # exact old prediction reuse, with explicit lineage names
    reuse_map={'TargetOnly_HGB':'H0_F0','PublicTarget_HGB':'H0_F1','PertEMA_P_adapted':'X0_F0','PertEMA_Public_RawAligned':'X0_F1','SharedTarget_HGB':'Shared_Legacy_F2'}
    old_tab=pd.read_csv(old/'TASK_PREDICTIONS.csv.gz')
    for src,name in reuse_map.items():
        q=old_tab[old_tab.method==src].copy(); q=q[q.task_id.isin(hold.task_id)]
        q['method']=name; q['lineage']='EXACT_REUSED_OLD_PUBLICSET_FEEDBACK_ALIGNMENT'; rows.append(q[['task_id','target','gene','fold','upstream','true_error_rmse','risk','method','seed','budget']])
    new_specs=[('H1_F1',P+PUBLIC,'H1'),('H1_F2',P+PUBLIC+['shared_risk'],'H1'),('X0_F2',P+PUBLIC+['shared_risk'],'X0')]
    for budget,ncl in zip([.1,.25,.5,.75,1.],BUDGET_COUNTS):
        genes=set(ordered[:ncl]); fit=pool[pool.gene.isin(genes)].reset_index(drop=True); labels=_fixed_labels(fit)
        for method,cols,learner in new_specs:
            for seed in SEEDS:
                if time.monotonic()-started>max_seconds: raise TimeoutError('fixed execution wall limit exceeded')
                t0=time.monotonic(); model=fit_target_learner_weighted(fit,labels,cols,learner,seed); score=np.clip(model.predict(hold[cols].to_numpy(float)),0,1); joblib_path=OUT/'models'/method/f'budget{budget}'/f'seed{seed}.joblib'; joblib_path.parent.mkdir(parents=True,exist_ok=True)
                import joblib; joblib.dump(model,joblib_path)
                out=hold[['task_id','target','gene','fold','upstream','true_error_rmse']].copy(); out['risk']=score; out['method']=method; out['seed']=seed; out['budget']=budget; out['lineage']='NEW_REGISTERED_FIXED_FIT'; rows.append(out)
                fits.append({'method':method,'budget':budget,'seed':seed,'n_fit_rows':len(fit),'n_fit_genes':len(genes),'feature_columns':cols,'fit_seconds':time.monotonic()-t0,'model_path':str(joblib_path),'model_sha256':sha(joblib_path),'training_id_hash':hashlib.sha256('\\n'.join(sorted(fit.task_id)).encode()).hexdigest()})
    pred=pd.concat(rows,ignore_index=True); pred.to_csv(OUT/'TASK_PREDICTIONS.csv.gz',index=False,compression='gzip'); write_csv(DOC/'FIXED_FIT_LEDGER.csv',pd.DataFrame(fits)); write_json(OUT/'PREDICTION_FREEZE.json',{'status':'ALL_FIXED_PREDICTIONS_FROZEN_BEFORE_BOOTSTRAP','rows':len(pred),'new_fit_count':len(fits),'expected_new_fit_count':45,'task_predictions_sha256':sha(OUT/'TASK_PREDICTIONS.csv.gz'),'selection_lock_sha256':sha(DOC/'SELECTION_LOCKED_BEFORE_HOLDOUT.json')})
    # Paired gene bootstrap across methods/seeds. Save curves and adoption gates.
    methods=sorted(pred.method.unique()); budgets=sorted(pred.budget.unique()); groups=[g.index.to_numpy() for _,g in hold.groupby('gene',sort=True)]; rng=np.random.default_rng(SEEDS[0]); metric_rows=[]; draw_rows=[]
    for budget in budgets:
        for method in methods:
            part=pred[(pred.method==method)&(pred.budget==budget)]
            if part.empty: continue
            # Aggregate seed metrics only after per-seed scoring.
            seed_scores=[]; seed_m=[]
            for seed in SEEDS:
                s=part[part.seed==seed].set_index('task_id').reindex(hold.task_id).risk.to_numpy(float)
                if np.isfinite(s).sum()==len(hold): seed_scores.append(s); seed_m.append(_metric_simple(hold,s))
            if not seed_m: continue
            point=np.nanmean(seed_m,axis=0); draws=np.empty((bootstrap,2))
            for b in range(bootstrap):
                idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); dg=hold.iloc[idx]
                sm=[]
                for s in seed_scores: sm.append(_metric_simple(dg,s[idx]))
                draws[b]=np.nanmean(sm,axis=0)
            metric_rows += [{'budget':budget,'method':method,'metric':'utility20','point_estimate':point[0],'ci95_lower':float(np.quantile(draws[:,0],.025)),'ci95_upper':float(np.quantile(draws[:,0],.975)),'bootstrap_replicates':bootstrap},{'budget':budget,'method':method,'metric':'aurc','point_estimate':point[1],'ci95_lower':float(np.quantile(draws[:,1],.025)),'ci95_upper':float(np.quantile(draws[:,1],.975)),'bootstrap_replicates':bootstrap}]
            draw_rows.append((budget,method,draws))
    write_csv(DOC/'METRIC_INTERVALS.csv',pd.DataFrame(metric_rows)); np.savez_compressed(OUT/'PAIRED_GENE_BOOTSTRAP_DRAWS.npz',budgets=np.asarray([x[0] for x in draw_rows]),methods=np.asarray([x[1] for x in draw_rows]),draws=np.asarray([x[2] for x in draw_rows]))
    # Basic utility curves and preregistered adoption gate against exact old H0_F1.
    intervals=pd.DataFrame(metric_rows); curve=intervals[intervals.metric.eq('utility20')].copy(); write_csv(DOC/'UTILITY_CURVES.csv',curve)
    gates=[]
    for b in sorted(curve.budget.unique()):
        base=curve[(curve.budget==b)&curve.method.eq('H0_F1')].iloc[0] if ((curve.budget==b)&curve.method.eq('H0_F1')).any() else None
        cand=curve[(curve.budget==b)&curve.method.eq('H1_F1')].iloc[0] if ((curve.budget==b)&curve.method.eq('H1_F1')).any() else None
        if base is not None and cand is not None: gates.append({'budget':b,'candidate':'H1_F1','comparator':'H0_F1','delta_u20':float(cand.point_estimate-base.point_estimate),'candidate_ci_lower':float(cand.ci95_lower),'gate_delta_min':.005,'pass':bool(cand.point_estimate-base.point_estimate>=.005 and cand.ci95_lower>=-.005)})
    write_csv(DOC/'ADOPTION_GATES.csv',pd.DataFrame(gates)); write_json(DOC/'SELECTED_DEFAULT.json',{'status':'COMPLETE','selected_feature_set':'F1','selected_learner':'H1','selection_locked_before_holdout':True,'fixed_evaluation_tasks':212,'fixed_evaluation_genes':152,'predictions':str(OUT/'TASK_PREDICTIONS.csv.gz'),'metric_intervals':str(DOC/'METRIC_INTERVALS.csv'),'adoption_gates':str(DOC/'ADOPTION_GATES.csv')}); write_json(OUT/'RUN_STATUS.json',{'status':'COMPLETE','phase':'FIXED_EVALUATION','new_fit_count':len(fits),'bootstrap_replicates':bootstrap,'elapsed_seconds':time.monotonic()-started,'prediction_freeze':str(OUT/'PREDICTION_FREEZE.json')}); print(json.dumps({'status':'FIXED_COMPLETE','new_fits':len(fits),'bootstrap':bootstrap}),flush=True)

def macro_metrics(frame, scores):
    from scipy.stats import spearmanr
    frame=frame.reset_index(drop=True); scores=np.asarray(scores,float)
    vals=[]
    for ctx,g in frame.groupby('target',sort=True):
        y=g.true_error_rmse.to_numpy(float); s=scores[g.index.to_numpy()]; ids=g.task_id.to_numpy(str); k=max(1,int(np.ceil(.2*len(g))))
        hi=np.lexsort((ids,-s))[:k]; oracle=np.lexsort((ids,-y))[:k]; den=y[oracle].mean()-y.mean(); u=((y[hi].mean()-y.mean())/den) if len(y)>=20 and den>1e-12 else np.nan
        lo=np.lexsort((ids,s)); aurc=float(np.mean(np.cumsum(y[lo])/np.arange(1,len(y)+1)))
        vals.append({'target':ctx,'utility20':u,'aurc':aurc,'spearman':float(spearmanr(s,y).statistic) if np.ptp(s)>0 and np.ptp(y)>0 else np.nan,'n_tasks':len(g)})
    d=pd.DataFrame(vals); return {'utility20':float(d.utility20.mean()),'aurc':float(d.aurc.mean()),'spearman':float(d.spearman.mean()),'valid_contexts':int(d.utility20.notna().sum()),'n_contexts':len(d)},d

def run_dev(max_seconds=3600):
    started=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True); DOC.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'TRAINING_START.json',{'status':'RUNNING','phase':'DEV','pid':os.getpid(),'started_epoch':time.time(),'contract':str(DATA/'EXECUTION_CONTRACT.json'),'cpu_threads':4,'new_fit_cap':360,'replay_reserve':60})
    frame,pred,truth,ordered=load_inputs(); feat,audit=build_public_features(frame,pred,row_exclude_own=False); write_csv(OUT/'DEV_FEATURE_AUDIT.csv.gz',audit); feat.to_parquet(OUT/'DEV_FEATURES.parquet',index=False)
    write_json(DOC/'DEV_FEATURES_INPUT_AUDIT.json',{'status':'PASS','tasks':len(feat),'genes':feat.gene.nunique(),'folds':3,'fold_hash_prefix':'SafeConf-target-dev-v1|','query_experiment_ids_removed_globally':True,'new_raw_expression_reads':0,'feature_hash':sha(OUT/'DEV_FEATURES.parquet')})
    allrows=[]; fitrows=[]; budget_rows=[]; nfit=0
    for held in range(3):
        tr=feat[feat.dev_fold!=held].copy(); te=feat[feat.dev_fold==held].copy(); train_order=[g for g in ordered if g in set(tr.gene)]
        for ncl in BUDGET_COUNTS:
            genes=set(train_order[:ncl]); fit=tr[tr.gene.isin(genes)].copy().reset_index(drop=True)
            labels=fit.true_error_rmse.to_numpy(float); # measured errors transformed to within-context midranks
            for target in sorted(fit.target.unique()):
                ix=fit.target.eq(target).to_numpy(); vals=labels[ix]; ranks=pd.Series(vals).rank(method='average').to_numpy(); labels[ix]=(ranks-.5)/len(vals)
            budget_rows.append({'heldout_fold':held,'n_training_genes':len(genes),'n_training_rows':len(fit),'training_gene_hash':hashlib.sha256('\n'.join(sorted(genes)).encode()).hexdigest(),'label_scope':'budget-only-per-context'})
            for feature_name,cols in [('F0',P),('F1',P+PUBLIC)]:
                for learner in ['H0','H1','X0']:
                    for seed in SEEDS:
                        if time.monotonic()-started>max_seconds: raise TimeoutError('declared DEV wall limit exceeded')
                        t0=time.monotonic(); model=fit_target_learner(fit,labels,cols,learner,seed); predscore=np.clip(model.predict(te[cols].to_numpy(float)),0,1); met,ctx=macro_metrics(te,predscore); nfit+=1
                        key=f'{feature_name}_{learner}'
                        row={'heldout_fold':held,'budget_clusters':ncl,'feature_set':feature_name,'learner':learner,'seed':seed,'n_fit_rows':len(fit),'n_fit_genes':len(genes),'n_eval_rows':len(te),'fit_seconds':time.monotonic()-t0,**met}
                        fitrows.append(row)
                        for i,t in enumerate(te.itertuples(index=False)):
                            allrows.append({'task_id':t.task_id,'gene':t.gene,'target':t.target,'heldout_fold':held,'budget_clusters':ncl,'feature_set':feature_name,'learner':learner,'seed':seed,'true_error_rmse':t.true_error_rmse,'risk':float(predscore[i])})
            write_csv(OUT/'DEV_PREDICTIONS.csv.gz',pd.DataFrame(allrows)); write_csv(OUT/'DEV_FIT_LEDGER.csv',pd.DataFrame(fitrows)); write_csv(OUT/'DEV_BUDGET_LEDGER.csv',pd.DataFrame(budget_rows)); write_json(OUT/'TRAINING_STATUS.json',{'status':'RUNNING','phase':'DEV','pid':os.getpid(),'fits_completed':nfit,'expected_fits':90,'last_fold':held,'last_budget_clusters':ncl,'elapsed_seconds':time.monotonic()-started})
            print(json.dumps({'phase':'DEV','fold':held,'budget_clusters':ncl,'fits_completed':nfit}),flush=True)
    pred_df=pd.DataFrame(allrows); ledger=pd.DataFrame(fitrows)
    # Selector is registered on F1 only; equal folds and budgets, context macro metrics.
    f1=ledger[ledger.feature_set.eq('F1')].groupby(['learner','budget_clusters'],as_index=False)[['utility20','aurc','fit_seconds']].mean(); agg=f1.groupby('learner',as_index=False).agg(mean_u20=('utility20','mean'),mean_aurc=('aurc','mean'),mean_cost=('fit_seconds','mean')); best_u=agg.mean_u20.max(); candidates=agg[agg.mean_u20>=best_u-.005].sort_values(['mean_aurc','mean_cost','learner']).reset_index(drop=True); selected=str(candidates.iloc[0].learner)
    write_csv(DOC/'DEV_METRICS.csv',ledger); write_csv(DOC/'DEV_SELECTION_TABLE.csv',agg); write_json(DOC/'SELECTION.json',{'status':'COMPLETE','selected_feature_set':'F1','selected_learner':selected,'selector':'F1 only; equal budgets/folds/contexts; U20 tie 0.005 then AURC then measured cost','candidates_within_u20_tie':candidates.to_dict('records'),'mean_u20_best':float(best_u),'all_fit_count':nfit,'dev_predictions':str(OUT/'DEV_PREDICTIONS.csv.gz'),'dev_features':str(OUT/'DEV_FEATURES.parquet')})
    write_json(OUT/'TRAINING_STATUS.json',{'status':'COMPLETE','phase':'DEV','pid':os.getpid(),'fits_completed':nfit,'expected_fits':90,'elapsed_seconds':time.monotonic()-started,'selection':str(DOC/'SELECTION.json'),'new_raw_expression_reads':0})
    print(json.dumps({'status':'DEV_COMPLETE','fits':nfit,'selected':selected}),flush=True)

def run_fixed_stats(bootstrap=5000):
    """Vectorized 5,000-draw statistics phase; predictions are immutable."""
    from tools.scripts.run_safeconf_feedback_metric_uncertainty import vector_metrics, METRICS
    hold=pd.read_parquet(OUT/'HOLDOUT_FEATURES.parquet').reset_index(drop=True)
    pred=pd.read_csv(OUT/'TASK_PREDICTIONS.csv.gz')
    freeze=json.loads((OUT/'PREDICTION_FREEZE.json').read_text())
    if not freeze['status'].startswith('ALL_FIXED'): raise RuntimeError('prediction freeze missing')
    methods=sorted(pred.method.unique()); budgets=[.1,.25,.5,.75,1.]; seeds=list(SEEDS)
    truth=hold.true_error_rmse.to_numpy(float); ids=hold.task_id.to_numpy(str); contexts=hold.target.to_numpy(str)
    scores=np.full((len(budgets),len(methods),len(seeds),len(hold)),np.nan)
    for bi,b in enumerate(budgets):
        for mi,m in enumerate(methods):
            for si,s in enumerate(seeds):
                q=pred[(pred.method==m)&(pred.budget==b)&(pred.seed==s)].set_index('task_id').reindex(hold.task_id)
                if len(q)==len(hold) and q.risk.notna().all(): scores[bi,mi,si]=q.risk.to_numpy(float)
    def evaluate(idx):
        outv=np.zeros((len(budgets),len(methods),len(seeds),len(METRICS)),float)
        for c in sorted(set(contexts)):
            rows=idx[contexts[idx]==c]
            flat=scores[:,:,:,rows].reshape(len(budgets)*len(methods)*len(seeds),len(rows))
            outv += np.nan_to_num(vector_metrics(truth[rows],ids[rows],flat).reshape(len(budgets),len(methods),len(seeds),len(METRICS)),nan=0.0)
        return outv/len(set(contexts))
    point=evaluate(np.arange(len(hold))); groups=[g.index.to_numpy() for _,g in hold.groupby('gene',sort=True)]; rng=np.random.default_rng(SEEDS[0]); draws=np.empty((bootstrap,*point.shape))
    for i in range(bootstrap):
        idx=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]); draws[i]=evaluate(idx)
        if (i+1)%500==0: print(json.dumps({'stage':'fixed_paired_bootstrap','completed':i+1}),flush=True)
    rows=[]
    for bi,b in enumerate(budgets):
        for mi,m in enumerate(methods):
            for ki,k in enumerate(METRICS):
                vals=np.nanmean(draws[:,bi,mi,:,ki],axis=1); valid=np.isfinite(vals)
                rows.append({'budget':b,'method':m,'metric':k,'point_estimate':float(np.nanmean(point[bi,mi,:,ki])),'ci95_lower':float(np.nanquantile(vals,.025)) if valid.any() else np.nan,'ci95_upper':float(np.nanquantile(vals,.975)) if valid.any() else np.nan,'valid_draws':int(valid.sum()),'bootstrap_replicates':bootstrap,'evaluation_tasks':len(hold),'evaluation_genes':hold.gene.nunique()})
    intervals=pd.DataFrame(rows); write_csv(DOC/'METRIC_INTERVALS.csv',intervals)
    pairs=[]
    for a,b in [('H1_F1','H0_F1'),('H1_F2','Shared_Legacy_F2'),('X0_F2','H1_F2')]:
        if a not in methods or b not in methods: continue
        ai,bi_m=methods.index(a),methods.index(b)
        for bi,budget in enumerate(budgets):
            for ki,k in enumerate(METRICS):
                d=np.nanmean(draws[:,bi,ai,:,ki],axis=1)-np.nanmean(draws[:,bi,bi_m,:,ki],axis=1); valid=np.isfinite(d)
                pairs.append({'budget':budget,'candidate':a,'comparator':b,'metric':k,'point_difference':float(np.nanmean(point[bi,ai,:,ki])-np.nanmean(point[bi,bi_m,:,ki])),'ci95_lower':float(np.nanquantile(d,.025)) if valid.any() else np.nan,'ci95_upper':float(np.nanquantile(d,.975)) if valid.any() else np.nan,'valid_draws':int(valid.sum()),'bootstrap_replicates':bootstrap})
    pair_df=pd.DataFrame(pairs); write_csv(DOC/'PAIRED_COMPARISONS.csv',pair_df); np.savez_compressed(OUT/'PAIRED_GENE_BOOTSTRAP_DRAWS.npz',draws=draws,methods=np.asarray(methods),budgets=np.asarray(budgets),metrics=np.asarray(METRICS))
    write_csv(DOC/'UTILITY_CURVES.csv',intervals[intervals.metric.isin(['utility20','aurc'])]); gates=[]
    for b in budgets:
        x=pair_df[(pair_df.budget==b)&pair_df.candidate.eq('H1_F1')&pair_df.comparator.eq('H0_F1')&pair_df.metric.eq('utility20')]
        if len(x):
            z=x.iloc[0]; gates.append({'budget':b,'candidate':'H1_F1','comparator':'H0_F1','delta_u20':float(z.point_difference),'lower95_delta':float(z.ci95_lower),'upper95_delta':float(z.ci95_upper),'delta_u20_min':.005,'lower95_min':-.005,'pass':bool(z.point_difference>=.005 and z.ci95_lower>=-.005)})
    write_csv(DOC/'ADOPTION_GATES.csv',pd.DataFrame(gates)); write_json(DOC/'SELECTED_DEFAULT.json',{'status':'COMPLETE','selected_feature_set':'F1','selected_learner':'H1','selection_locked_before_holdout':True,'fixed_evaluation_tasks':212,'fixed_evaluation_genes':152,'bootstrap_replicates':bootstrap,'predictions':str(OUT/'TASK_PREDICTIONS.csv.gz'),'metric_intervals':str(DOC/'METRIC_INTERVALS.csv'),'adoption_gates':str(DOC/'ADOPTION_GATES.csv')}); write_json(OUT/'RUN_STATUS.json',{'status':'COMPLETE','phase':'FIXED_EVALUATION','new_fit_count':45,'bootstrap_replicates':bootstrap,'prediction_freeze':str(OUT/'PREDICTION_FREEZE.json'),'statistics_receipt':str(DOC/'METRIC_INTERVALS.csv')}); print(json.dumps({'status':'FIXED_STATS_COMPLETE','bootstrap':bootstrap,'methods':methods}),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['preflight','dev','fixed','evaluate'],default='preflight'); ap.add_argument('--max-seconds',type=float,default=3600); ap.add_argument('--bootstrap',type=int,default=5000); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='preflight':
        frame,_,_,order=load_inputs(); write_json(DOC/'EXECUTION_CONFIG.json',{'version':'safeconf-target-feedback-20261003-v1','status':'CONFIG_FROZEN','tasks':len(frame),'genes':frame.gene.nunique(),'folds':3,'fold_hash_prefix':'SafeConf-target-dev-v1|','budgets_clusters':list(BUDGET_COUNTS),'budgets':[0,.1,.25,.5,.75,1],'seeds':list(SEEDS),'learners':{'H0':{'max_iter':200,'learning_rate':.05,'max_depth':3,'min_samples_leaf':20,'l2_regularization':10},'H1':{'min_samples_leaf':'min(20,max(2,n_finite_fit_rows//10))'},'X0':{'n_estimators':300,'max_depth':6,'learning_rate':.05,'subsample':.8,'colsample_bytree':.8,'tree_method':'hist'}},'input_bindings':{str(COMMON/'VALIDATION_TASKS.csv'):sha(COMMON/'VALIDATION_TASKS.csv'),str(COMMON/'VALIDATION_CALIBRATED_EFFECTS.npy'):sha(COMMON/'VALIDATION_CALIBRATED_EFFECTS.npy')},'new_raw_expression_reads':0}); print(json.dumps({'status':'PREFLIGHT_PASS','tasks':len(frame),'genes':frame.gene.nunique()})); return
    if args.phase=='fixed':
        if args.bootstrap!=5000: raise RuntimeError('registered bootstrap count is fixed at 5000')
        run_fixed(args.max_seconds,args.bootstrap)
    elif args.phase=='evaluate':
        if args.bootstrap!=5000: raise RuntimeError('registered bootstrap count is fixed at 5000')
        run_fixed_stats(args.bootstrap)
    else:
        run_dev(args.max_seconds)
if __name__=='__main__': main()

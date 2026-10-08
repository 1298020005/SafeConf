#!/usr/bin/env python3
"""Executable SafeConf v2.1 work packages, with independent immutable outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tools.safeconf_continual.research import P, PUBLIC, rank_labels, cluster_weights
from tools.safeconf_continual.submission_evidence import (
    BUDGETS, REVIEWS, ORDERS, MODEL_SEEDS, NULL_SEEDS, sha, digest_ids,
    write_json, ordered_genes, endpoint_arrays, point_metrics, ClusterBootstrap, summarize_draws,
)
from tools.scripts.run_safeconf_pertema_current_truth_v1 import (
    POOL, HOLD, NATIVE, NATIVE_FEATURES, OFFICIAL_ROOT, OFFICIAL_COMMIT, official_gbt,
)

RUN = Path('/home/yyf/runtime_artifacts/safeconf_submission_evidence_20261009_v21')
COMMON = Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
SAMS = Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/sams_v1')
DOC = ROOT/'docs/研究推进/20261009_投稿证据_v21'
CONFIG = {
    'run_id': RUN.name, 'baseline_commit': '280f631', 'contract': 'v2.1',
    'feedback_budgets': BUDGETS, 'review_budgets': REVIEWS, 'feedback_order_seeds': ORDERS,
    'learner_seeds': MODEL_SEEDS, 'content_permutation_seeds': NULL_SEEDS,
    'bootstrap_replicates': 5000, 'bootstrap_unit': 'perturbation_gene_cluster',
    'algorithm_runs_are_biological_samples': False, 'risk_direction': 'higher is more dangerous',
    'primary_endpoint': 'delta_rmse', 'secondary_endpoints': ['pearson_error', 'effect_top200_rmse'],
    'label_equivalent_margin': -.005, 'label_equivalent_ci': 'paired_cluster_95_lower',
    'multiplicity_sensitivity': 'five-budget Bonferroni, per-budget 99% intervals',
    'competence_gate': {'mean_relative_rmse_max': .02, 'ci95_upper_max': .02,
        'noninferior_strata_min': .6, 'valid_strata_min': .8, 'predicted_variance_ratio_min': .01},
    'external_split': {'predictor_train': .5, 'feedback': .25, 'confirmation': .25,
        'hash_prefix': 'SafeConf-Gladstone-v21-split'},
    'external_predictors': {'ridge': {'alpha': 100}, 'mlp': {'hidden': 256, 'epochs': 30,
        'lr': .001, 'batch_size': 2048, 'device': 'cpu', 'per_context_training': True}},
    'public_default': 'PublicRule', 'target_development_preparation': {'rows': 542, 'genes': 377},
    'zero_target_error_supervision_scope': 'risk layer fitting/selection/calibration, upstream costs separate',
    'paper_pdf': 'PAUSED', 'protected_pids': [1873824, 2428149],
    'resources': {'threads': 4, 'memory_target_GiB': 16, 'new_large_upstream': 0,
        'gpu_training': 0, 'original_cumulative_download_cap_bytes': 100_000_000_000,
        'original_cumulative_gpu_cap_hours': 96},
}


def status(package, value):
    write_json(RUN/package/'STATUS.json', value)
    p = RUN/'RUN_STATE.json'
    state = json.loads(p.read_text()) if p.exists() else {'research_complete': False}
    state[package] = value
    write_json(p, state)


def preflight():
    RUN.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    path = RUN/'CONFIG.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(CONFIG)):
        raise RuntimeError('different configuration at this run_id')
    write_json(path, CONFIG); write_json(DOC/'CONFIG.json', CONFIG)
    pool, hold = pd.read_parquet(POOL), pd.read_parquet(HOLD)
    assert len(pool)==331 and pool.gene.nunique()==228 and len(hold)==212 and hold.gene.nunique()==152
    assert not set(pool.gene)&set(hold.gene) and pool.task_id.is_unique and hold.task_id.is_unique
    official = subprocess.check_output(['git','-C',str(OFFICIAL_ROOT),'rev-parse','HEAD'],text=True).strip()
    if official != OFFICIAL_COMMIT: raise RuntimeError('PertEMA source version changed')
    files = [POOL,HOLD,NATIVE,COMMON/'public_mcfaline_trainval/public_memory.parquet',
        COMMON/'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy',
        SAMS/'DecoderOnly_RISK_FEATURES.parquet',SAMS/'SAMS_VAE_RISK_FEATURES.parquet']
    assets=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    receipt = {'status':'PASS','assets':assets,'official_commit':official,
        'pool':{'rows':331,'genes':228},'evaluation':{'rows':212,'genes':152,'role':'SEEN'},
        'protected_pids_alive':{str(p):Path(f'/proc/{p}').exists() for p in CONFIG['protected_pids']},
        'unrelated_worktree_files_not_staged':True,'permanent_test_truth_read':False,
        'own_script_sha256':sha(Path(__file__)), 'config_sha256':sha(path)}
    write_json(RUN/'PREFLIGHT_MANIFEST.json',receipt);write_json(DOC/'PREFLIGHT_MANIFEST.json',receipt)
    status('preflight', receipt)
    print(json.dumps({'preflight':'PASS','run_id':RUN.name,'pool':331,'evaluation':212}),flush=True)


def join_native(frame):
    columns=['task_id']+NATIVE_FEATURES
    native=pd.read_parquet(NATIVE,columns=columns).set_index('task_id')
    f=frame.join(native[NATIVE_FEATURES],on='task_id',rsuffix='_old_native').copy()
    f['native_prediction_abs_mean']=f.prediction_abs_mean
    if not frame.task_id.isin(native.index).all(): raise RuntimeError('native task identity coverage incomplete')
    # A missing control-gene value is a legitimate XGBoost native missing
    # feature. It is not a missing task, and is shared by both input arms.
    return f


def native_x(train, query, columns):
    xt, xq = train[columns].to_numpy(float), query[columns].to_numpy(float)
    if 'native_training_similarity' not in columns: return xt,xq
    ei=[i for i,c in enumerate(columns) if c.startswith('native_embedding_')]
    si=columns.index('native_training_similarity')
    et, eq=xt[:,ei],xq[:,ei]
    good=np.isfinite(et).all(1);qgood=np.isfinite(eq).all(1)
    proto, owner=et[good],train.gene.astype(str).to_numpy()[good]
    xt[:,si]=np.nan;xq[:,si]=np.nan
    if len(proto):
        dt=cdist(et[good],proto);dt[owner[:,None]==owner[None,:]]=np.inf
        value=dt.min(1);value[~np.isfinite(value)]=np.nan;xt[good,si]=value
        xq[qgood,si]=cdist(eq[qgood],proto).min(1)
    return xt,xq


def selected_support(frame):
    memory=pd.read_parquet(COMMON/'public_mcfaline_trainval/public_memory.parquet')
    by_gene=memory.groupby('perturbation_target').indices
    forbidden=set('McFaline23::'+frame.task_id.astype(str));values=[]
    for row in frame.itertuples(index=False):
        m=memory.iloc[np.asarray(by_gene.get(str(row.gene),[]),int)]
        m=m[~m.experiment_id.isin(forbidden)&~((m.context.astype(str)==str(row.context))&(m.condition.astype(str)==str(row.treatment)))]
        same=m.context.astype(str).eq(str(row.context))
        m=m[same] if same.any() else m
        values.append(float(np.log1p(m.n_cells.sum())) if len(m) else np.nan)
    return np.asarray(values)


def main_endpoints(model, query):
    tasks=pd.read_csv(COMMON/'TEST_TASKS.csv').reset_index().set_index('task_id')
    rows=tasks.loc[query.task_id,'index'].to_numpy(int)
    pred_path=COMMON/'TEST_CALIBRATED_EFFECTS.npy' if model=='DecoderOnly' else SAMS/'TEST_PREDICTED_EFFECTS.npy'
    # All these tasks were already legitimately opened in prior experiments.
    p=np.asarray(np.load(pred_path,mmap_mode='r')[rows],float)
    y=np.asarray(np.load(COMMON/'TEST_TRUE_EFFECTS.npy',mmap_mode='r')[rows],float)
    errors=endpoint_arrays(p,y)
    if not np.allclose(errors['delta_rmse'],query.true_error_rmse,rtol=1e-5,atol=1e-8):
        raise RuntimeError(f'{model} current error contract mismatch')
    return errors


def evidence_fit(raw_rmse=False, support_diagnostic=False):
    package='support_information_diagnostic' if support_diagnostic else 'evidence_budget_raw_rmse' if raw_rmse else 'evidence_budget'
    out=RUN/package;out.mkdir(parents=True,exist_ok=True)
    if (out/'PREDICTIONS.parquet').exists():
        print('reusing completed frozen budget predictions',flush=True);return
    pool0,hold0=pd.read_parquet(POOL),pd.read_parquet(HOLD)
    model_files={'DecoderOnly':SAMS/'DecoderOnly_RISK_FEATURES.parquet','SAMS_VAE':SAMS/'SAMS_VAE_RISK_FEATURES.parquet'}
    started,cpu=time.monotonic(),time.process_time();parts=[];ledger=[];cdf_records=[];fit_count=0
    status('evidence_budget',{'status':'FITTING','pid':os.getpid(),'fits':0})
    for predictor,path in model_files.items():
        whole=pd.read_parquet(path).set_index('task_id')
        pool=join_native(whole.loc[pool0.task_id].reset_index());query=join_native(whole.loc[hold0.task_id].reset_index())
        pool['selected_support_log']=selected_support(pool);query['selected_support_log']=selected_support(query)
        assert np.allclose(query.simple_history_risk.to_numpy()**2,
                           query.prediction_prior_rmse.to_numpy()**2+query.prior_uncertainty.to_numpy()**2,rtol=1e-5,atol=1e-8)
        query[['task_id','gene','target','context','treatment','true_error_rmse','simple_history_risk',
               'predicted_magnitude','support_risk','selected_support_log']].to_parquet(out/f'{predictor}_TASKS.parquet',index=False)
        ep=main_endpoints(predictor,query);np.savez(out/f'{predictor}_ENDPOINTS.npz',**ep)
        fsets={'Target_P6':P,'PertEMA_Native61':NATIVE_FEATURES,
               'PertEMA_Native61_Public':NATIVE_FEATURES+PUBLIC}
        if raw_rmse:
            fsets={k+'_RawRMSE':v for k,v in fsets.items()}
        if support_diagnostic:
            fsets={'Native61_SelectedSupport_RawRMSE':NATIVE_FEATURES+['selected_support_log'],
                   'Native61_Public_SelectedSupport_RawRMSE':NATIVE_FEATURES+PUBLIC+['selected_support_log']}
        cache={}
        for order_seed in ORDERS:
            ordered=ordered_genes(pool,order_seed)
            for budget in BUDGETS[1:]:
                genes=ordered[:math.ceil(budget*len(ordered))]
                train=pool[pool.gene.isin(genes)].copy().reset_index(drop=True)
                labels,audit=rank_labels(train,f'v21/{predictor}/{order_seed}/{budget}',budget)
                cdf_records.extend(audit);valid=np.isfinite(labels)
                if raw_rmse:
                    labels=train.true_error_rmse.to_numpy(float);valid=np.isfinite(labels)
                if valid.sum()<2: raise RuntimeError('insufficient budget-only CDF labels')
                for method,cols in fsets.items():
                    xt,xq=native_x(train,query,cols)
                    for seed in MODEL_SEEDS:
                        cache_key=(digest_ids(train.task_id),method,seed)
                        reused=cache_key in cache
                        if not reused:
                            model=official_gbt(seed)
                            model.fit(xt[valid],labels[valid],sample_weight=cluster_weights(train.loc[valid]))
                            risk=np.asarray(model.predict(xq),float)
                            if not np.isfinite(risk).all(): raise RuntimeError('nonfinite frozen risk')
                            cache[cache_key]=risk;fit_count+=1
                            model.get_booster().save_model(out/f'model_{predictor}_{method}_o{order_seed}_b{budget}_s{seed}.json')
                        risk=cache[cache_key]
                        part=query[['task_id']].copy();part['predictor']=predictor;part['method']=method
                        part['feedback_budget']=budget;part['order_seed']=order_seed;part['learner_seed']=seed
                        part['risk_score']=risk;parts.append(part)
                        ledger.append({'predictor':predictor,'method':method,'feedback_budget':budget,
                            'order_seed':order_seed,'learner_seed':seed,'feedback_genes':len(genes),
                            'feedback_rows':len(train),'finite_training_rows':int(valid.sum()),
                            'error_record_union_hash':digest_ids(train.task_id),'reuse_identical_fit':reused,
                            'cdf_error_rows':len(train),'risk_training_error_rows':int(valid.sum()),
                            'target_preparation_rows':542,'target_preparation_genes':377,
                            'target_evaluation_error_rows_used_for_fitting':0,
                            'label_target':'raw current RMSE' if raw_rmse else 'budget-only compatible-group error CDF',
                            'calibration':'NONE; raw ranking, not complete conformal reproduction',
                            'similarity':'exclude same biological gene in fitting rows; training prototypes for query'})
                status('evidence_budget',{'status':'FITTING','pid':os.getpid(),'fits':fit_count,
                    'predictor':predictor,'order_seed':order_seed,'budget':budget})
            print(json.dumps({'package':'evidence_budget','predictor':predictor,'order_seed':order_seed,
                              'fits_completed':fit_count,'wall_seconds':time.monotonic()-started}),flush=True)
    predictions=pd.concat(parts,ignore_index=True)
    predictions.to_parquet(out/'PREDICTIONS.parquet',index=False)
    pd.DataFrame(ledger).to_csv(out/'INFORMATION_BUDGET_LEDGER.csv',index=False)
    pd.DataFrame(cdf_records).to_csv(out/'ERROR_LABEL_CDF_AUDIT.csv',index=False)
    write_json(out/'PREDICTION_FREEZE.json',{'prediction_sha256':sha(out/'PREDICTIONS.parquet'),
        'config_sha256':sha(RUN/'CONFIG.json'),'fits':fit_count,'new_gpu_hours':0,
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
        'default':'PublicRule','role':'SEEN same-task descriptive comparison, no holdout selection',
        'labels':'raw current RMSE, official error-regression recipe adaptation' if raw_rmse else 'budget-only compatible-group midrank CDF; same labels/weights across three XGB inputs'})
    status('evidence_budget',{'status':'PREDICTIONS_FROZEN_STATISTICS_NEXT','fits':fit_count,
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu})


def evidence_statistics(raw_rmse=False, support_diagnostic=False):
    package='support_information_diagnostic' if support_diagnostic else 'evidence_budget_raw_rmse' if raw_rmse else 'evidence_budget'
    out=RUN/package
    pred=pd.read_parquet(out/'PREDICTIONS.parquet')
    rows=[];paired=[];label_equivalent=[]
    for predictor in pred.predictor.unique():
        f=pd.read_parquet(out/f'{predictor}_TASKS.parquet');errors=dict(np.load(out/f'{predictor}_ENDPOINTS.npz'))
        ids=f.task_id.to_numpy(str);public=f.simple_history_risk.to_numpy(float)
        baselines={'Magnitude':f.predicted_magnitude.to_numpy(float),'HistorySupport':f.support_risk.to_numpy(float),
                   'PublicRule':public}
        if 'selected_support_log' not in f:f['selected_support_log']=selected_support(f)
        baselines['HistorySupport_selected']=-f.selected_support_log.to_numpy(float)
        cross=pd.read_csv(SAMS/'CROSSFAMILY_TASK_PREDICTIONS.csv.gz')
        line='SAMS_VAE_to_DecoderOnly' if predictor=='DecoderOnly' else 'DecoderOnly_to_SAMS_VAE'
        source=cross[(cross.line==line)&(cross.method=='Source_hgb')].set_index('task_id').loc[ids]
        if not np.allclose(source.true_error_rmse,f.true_error_rmse,rtol=1e-6,atol=1e-9):
            raise RuntimeError('frozen cross-family Source baseline has different target error contract')
        baselines['SourceRisk_frozen']=source.risk.to_numpy(float)
        runs={}
        for key,q in pred[pred.predictor.eq(predictor)].groupby(['method','feedback_budget','order_seed','learner_seed'],sort=True):
            runs[key]=q.set_index('task_id').loc[ids].risk_score.to_numpy(float)
        for endpoint,e in errors.items():
            for method,s in baselines.items():
                for review in REVIEWS:
                    for scope,use in [('global',np.arange(len(f)))]+[(t,np.asarray(v,int)) for t,v in f.groupby('target').indices.items()]:
                        rows.append({'predictor':predictor,'endpoint':endpoint,'method':method,'feedback_budget':0,
                            'order_seed':-1,'learner_seed':-1,'scope':scope,**point_metrics(e[use],s[use],ids[use],review)})
            for (method,budget,order,seed),s in runs.items():
                for review in REVIEWS:
                    for scope,use in [('global',np.arange(len(f)))]+[(t,np.asarray(v,int)) for t,v in f.groupby('target').indices.items()]:
                        rows.append({'predictor':predictor,'endpoint':endpoint,'method':method,'feedback_budget':budget,
                            'order_seed':order,'learner_seed':seed,'scope':scope,**point_metrics(e[use],s[use],ids[use],review)})
        table=pd.DataFrame(rows)
        engine=ClusterBootstrap(f,errors['delta_rmse'])
        def macro(s):
            return np.nanmean([point_metrics(errors['delta_rmse'][ix],s[ix],ids[ix])['utility']
                               for ix in f.groupby('target').indices.values()])
        public_point=macro(public)
        boot_cache={}
        for method in sorted({k[0] for k in runs}):
            for budget in BUDGETS[1:]:
                scores=np.asarray([s for k,s in runs.items() if k[0]==method and k[1]==budget])
                point=float(np.mean([macro(s) for s in scores])-public_point)
                # Identical rankings from repeated full-budget orders share computation,
                # but retain their actual multiplicities in the algorithm mean.
                unique,inv=np.unique(np.asarray([np.lexsort((ids,-s)) for s in scores]),axis=0,return_inverse=True)
                ub=[]
                for order in unique:
                    representative=np.empty(len(ids));representative[order]=np.arange(len(ids),0,-1)
                    key=tuple(order.tolist())
                    if key not in boot_cache: boot_cache[key]=engine.utility(representative)
                    ub.append(boot_cache[key])
                draw=np.mean(np.asarray(ub)[inv],axis=0)-engine.utility(public)
                record={'predictor':predictor,'endpoint':'delta_rmse','comparison':method+'-minus-PublicRule',
                    'method':method,'feedback_budget':budget,'n_tasks':len(f),'n_clusters':f.gene.nunique(),
                    'n_algorithm_runs':len(scores),**summarize_draws(draw,point)}
                paired.append(record)
                print(json.dumps({'statistics':predictor,'method':method,'budget':budget,
                    'delta':point,'ci95':[record['ci95_lower'],record['ci95_upper']]}),flush=True)
            p=[r for r in paired if r['predictor']==predictor and r['method']==method]
            for kind,key in [('primary_nominal_95','ci95_lower'),('sensitivity_five_budget_bonferroni','ci99_lower')]:
                passing=[r for r in p if r[key]>=-.005]
                b=min(r['feedback_budget'] for r in passing) if passing else None
                led=pd.read_csv(out/'INFORMATION_BUDGET_LEDGER.csv')
                sel=led[(led.predictor==predictor)&(led.method==method)&(led.feedback_budget==(b if b is not None else 1.))]
                label_equivalent.append({'predictor':predictor,'target_method':method,'definition':kind,
                    'reference':'PublicRule zero target-error risk training','budget':b,
                    'right_censored':b is None,'tested_budget_upper':1.,
                    'error_rows_min':int(sel.feedback_rows.min()),'error_rows_median':float(sel.feedback_rows.median()),
                    'error_rows_max':int(sel.feedback_rows.max()),'gene_clusters':int(sel.feedback_genes.iloc[0]),
                    'preparation_cost_rows_separate':542,'aurc_is_definition_condition':False,
                    'ci_conditions_on_fitted_algorithm_runs':True})
    table=pd.DataFrame(rows);table.to_csv(out/'ALL_METRICS.csv',index=False)
    # Equal-context macro first, then algorithm means; avoid seed/row pooling.
    numeric=['utility','aurc','spearman','high_error_found','high_error_precision','high_error_recall',
             'high_risk_miss_rate','remaining_mean_error']
    strata=table[table.scope.ne('global')]
    group=['predictor','endpoint','method','feedback_budget','review_fraction','order_seed','learner_seed']
    macro=strata.groupby(group,as_index=False)[numeric].mean();macro.to_csv(out/'MACRO_PER_RUN.csv',index=False)
    macro.groupby(group[:-2],as_index=False)[numeric].agg(['mean','median','min','max']).reset_index().to_csv(out/'CURVE_SUMMARY.csv',index=False)
    macro.groupby(group[:-1],as_index=False)[numeric].mean().to_csv(out/'FEEDBACK_ORDER_STABILITY.csv',index=False)
    macro.groupby(group[:-2]+['learner_seed'],as_index=False)[numeric].mean().to_csv(out/'LEARNER_SEED_STABILITY.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'BIOLOGICAL_CLUSTER_BOOTSTRAP.csv',index=False)
    pd.DataFrame(label_equivalent).to_csv(out/'LABEL_EQUIVALENT_BUDGET.csv',index=False)
    status(package,{'status':'COMPLETE','runs':pred.groupby(['predictor','method','feedback_budget','order_seed','learner_seed']).ngroups,
        'label_equivalent':label_equivalent,'default':'PublicRule','frozen_holdout_is_SEEN':True})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['preflight','budget-fit','budget-statistics','budget','budget-raw-rmse','support-diagnostic'],required=True)
    a=ap.parse_args()
    if a.phase=='preflight':preflight()
    else:
        if not (RUN/'PREFLIGHT_MANIFEST.json').exists():preflight()
        if a.phase in ('budget-fit','budget'):evidence_fit()
        if a.phase in ('budget-statistics','budget'):evidence_statistics()
        if a.phase=='budget-raw-rmse':evidence_fit(raw_rmse=True);evidence_statistics(raw_rmse=True)
        if a.phase=='support-diagnostic':evidence_fit(raw_rmse=True,support_diagnostic=True);evidence_statistics(raw_rmse=True,support_diagnostic=True)


if __name__=='__main__':main()

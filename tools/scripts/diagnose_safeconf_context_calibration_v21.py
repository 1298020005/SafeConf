#!/usr/bin/env python3
"""Separate within-context retrieval from calibration of a whole-cohort rank.

No primary endpoint or adopted default changes. Add a label-matched trivial
state-mean comparator so a learned context offset cannot masquerade as gene
specific failure learning. All models/scores already generated are reused.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import sys,math,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN,POOL,HOLD,SAMS
from tools.safeconf_continual.research import cluster_weights
from tools.safeconf_continual.submission_evidence import (write_json,point_metrics,ordered_genes,ClusterBootstrap,
    summarize_draws,BUDGETS,ORDERS)


def main():
    out=RUN/'context_calibration_diagnostic';out.mkdir(exist_ok=True)
    write_json(out/'EXPERIMENT_CARD.json',{'trigger':'Native+Public improves whole-cohort review despite limited macro increment',
        'role':'SEEN diagnostic, no evaluation-selected deployment','primary_context_macro_label_equivalent_unchanged':True,
        'secondary_endpoint':'global U20 at20% review; exact same full cohort','simple_baseline':'budget-only Target state mean of raw RMSE',
        'source_diagnostic':'other qualified predictor context mean error; not a new Source feature learner',
        'new_model_fits':0,'orders':list(ORDERS),'bootstrap_genes':5000})
    main=RUN/'evidence_budget_raw_rmse';pred=pd.read_parquet(main/'PREDICTIONS.parquet')
    pool_ids=pd.read_parquet(POOL).task_id.tolist();metrics=[];stats=[];labels=[]
    for name in ['DecoderOnly','SAMS_VAE']:
        whole=pd.read_parquet(SAMS/f'{name}_RISK_FEATURES.parquet').set_index('task_id')
        pool=whole.loc[pool_ids].reset_index();q=pd.read_parquet(main/f'{name}_TASKS.parquet')
        ids=q.task_id.astype(str).to_numpy();y=q.true_error_rmse.to_numpy();public=q.simple_history_risk.to_numpy();engine=ClusterBootstrap(q,y)
        def state_key(f):return f.context.astype(str)+'::'+f.treatment.astype(str)
        pool['state']=state_key(pool);query_state=state_key(q)
        state_scores={b:[] for b in BUDGETS[1:]}
        for order in ORDERS:
            ordered=ordered_genes(pool,order)
            for budget in BUDGETS[1:]:
                train=pool[pool.gene.isin(ordered[:math.ceil(budget*len(ordered))])].copy()
                train['w']=cluster_weights(train);overall=np.average(train.true_error_rmse,weights=train.w)
                means={s:float(np.average(t.true_error_rmse,weights=t.w)) for s,t in train.groupby('state')}
                score=query_state.map(means).fillna(overall).to_numpy(float);state_scores[budget].append(score)
                metrics.append({'predictor':name,'method':'Target_state_mean','feedback_budget':budget,'order_seed':order,
                    'training_error_rows':len(train),'training_gene_clusters':train.gene.nunique(),
                    **point_metrics(y,score,ids)})
        runs={}
        for (method,budget,order,seed),t in pred[pred.predictor.eq(name)].groupby(['method','feedback_budget','order_seed','learner_seed']):
            risk=t.set_index('task_id').loc[ids].risk_score.to_numpy();runs.setdefault((method,budget),[]).append(risk)
        for b,v in state_scores.items():runs[('Target_state_mean',b)]=v
        cache={};reference=engine.utility(public,macro=False);p=point_metrics(y,public,ids)['utility']
        for (method,budget),vectors in runs.items():
            draws=[]
            for risk in vectors:
                order=tuple(np.lexsort((ids,-risk)))
                if order not in cache:cache[order]=engine.utility(risk,macro=False)
                draws.append(cache[order])
            delta=np.mean([point_metrics(y,r,ids)['utility'] for r in vectors])-p
            stats.append({'predictor':name,'method':method,'feedback_budget':budget,'scope':'SECONDARY_GLOBAL_RANK',
                **summarize_draws(np.mean(draws,axis=0)-reference,delta)})
        for method in sorted(set(k[0] for k in runs)):
            for definition,col in [('nominal95','ci95_lower'),('five_budget_sensitivity99','ci99_lower')]:
                eligible=[s for s in stats if s['predictor']==name and s['method']==method and s[col]>=-.005]
                b=min(s['feedback_budget'] for s in eligible) if eligible else None
                labels.append({'predictor':name,'method':method,'endpoint_scope':'SECONDARY_GLOBAL_RANK',
                    'definition':definition,'label_equivalent_budget':b,'right_censored':b is None,
                    'does_not_replace_registered_primary':True})
    pd.DataFrame(metrics).to_csv(out/'TARGET_STATE_MEAN_METRICS.csv',index=False)
    pd.DataFrame(stats).to_csv(out/'GLOBAL_BUDGET_PAIRED_BOOTSTRAP.csv',index=False)
    pd.DataFrame(labels).to_csv(out/'GLOBAL_LABEL_EQUIVALENT_SENSITIVITY.csv',index=False)
    write_json(out/'STATUS.json',{'status':'COMPLETE','scope':'SEEN secondary diagnosis','new_fits':0,'default_changed':False})

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Actual whole-cohort review effects, distinct from context-macro U20."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN
from tools.safeconf_continual.submission_evidence import ClusterBootstrap,point_metrics,summarize_draws,write_json


def actual_draws(engine,score):
    """Count-weight identical to drawing whole genes and then reviewing 20%."""
    e=engine.error;w=engine.weights.astype(np.int32);n=w.sum(1);k=np.ceil(.2*n).astype(int)
    def selected(order):
        cw=w[:,order];prior=np.cumsum(cw,axis=1)-cw;take=np.clip(k[:,None]-prior,0,cw)
        aligned=np.empty_like(take);aligned[:,order]=take
        return aligned
    take=selected(np.lexsort((engine.ids,-np.asarray(score))))
    severe=selected(np.lexsort((engine.ids,-e)))
    found=np.minimum(take,severe).sum(1)
    remaining=((w-take)@e)/(n-k)
    return {'high_error_found':found,'remaining_mean_error':remaining,
        'high_error_recall':found/k,'utility_global':engine.utility(score,macro=False)}


def main():
    rows=[]
    for predictor in ['DecoderOnly','SAMS_VAE']:
        root=RUN/'evidence_budget_raw_rmse';f=pd.read_parquet(root/f'{predictor}_TASKS.parquet')
        s=pd.read_parquet(root/f'{predictor}_STRONG_BASELINE_SCORES.parquet').set_index('task_id').loc[f.task_id]
        engine=ClusterBootstrap(f,f.true_error_rmse)
        public=actual_draws(engine,s.PublicRule);p=point_metrics(f.true_error_rmse,s.PublicRule,f.task_id)
        for reference in ['Magnitude','HistorySupport_all','HistorySupport_selected','SourceRisk_frozen']:
            base=actual_draws(engine,s[reference]);b=point_metrics(f.true_error_rmse,s[reference],f.task_id)
            for endpoint in ['high_error_found','remaining_mean_error','high_error_recall','utility_global']:
                field='utility' if endpoint=='utility_global' else endpoint
                rows.append({'predictor':predictor,'reference':reference,'endpoint':endpoint,'scope':'GLOBAL_SINGLE_RANKING',
                    'point_Public':p[field],'point_reference':b[field],
                    **summarize_draws(public[endpoint]-base[endpoint],p[field]-b[field])})
    pd.DataFrame(rows).to_csv(RUN/'GLOBAL_REVIEW_PAIRED_BOOTSTRAP.csv',index=False)
    write_json(RUN/'GLOBAL_REVIEW_SEMANTICS.json',{'review_budget':.2,'point_review_count':43,'point_tasks':212,
        'biological_gene_clusters':152,'replicates':5000,'macro_primary_label_equivalent_unchanged':True,
        'bootstrap_review_count':'ceil(0.2 * resampled task copies)','algorithm_seeds_not_biological_units':True,
        'full_ranking_scope_is_not_context_macro':'both reported separately','new_model_fits':0})

if __name__=='__main__':main()

#!/usr/bin/env python3
"""One DEV-only affine error-unit diagnostic of the fixed XGBoost recipe.

Raw RMSE trees often stop at leaves while CDF-target trees continue splitting.
Test a reversible training-side change of units, retaining the raw error
ordering, model, features, feedback identities, and loss family.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.scripts.run_safeconf_submission_evidence_v21 import RUN,join_native,native_x,official_gbt,NATIVE_FEATURES
from tools.safeconf_continual.research import cluster_weights
from tools.safeconf_continual.submission_evidence import write_json,ordered_genes,digest_ids,point_metrics,ClusterBootstrap,summarize_draws


def main():
    out=RUN/'target_error_units_dev';out.mkdir(exist_ok=True)
    if (out/'DECISION.json').exists():return
    frame=pd.read_parquet('/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/DEV_FEATURES.parquet')
    controls=np.load('/home/yyf/runtime_artifacts/safeconf_research_20261001/native_control_reference/CONTROL_REFERENCE.npz')
    gene_map={g:i for i,g in enumerate(controls['genes'])};state_map={s:i for i,s in enumerate(controls['states'])}
    gi=np.asarray([gene_map.get(g,-1) for g in frame.gene]);state=frame.context.astype(str)+'::'+frame.condition.astype(str)
    si=np.asarray([state_map.get(s,-1) for s in state]);good=(gi>=0)&(si>=0)
    frame['native_prediction_abs_mean']=frame.prediction_abs_mean
    for column,key in [('native_control_baseline','baseline'),('native_control_dropout','dropout'),('native_control_plate_variance_proxy','plate_variance')]:
        values=np.full(len(frame),np.nan);values[good]=controls[key][si[good],gi[good]];frame[column]=values
    for k in range(6):frame[f'native_state_{k}']=(si==k).astype(float)
    for k in range(50):
        values=np.full(len(frame),np.nan);values[gi>=0]=controls['embedding'][gi[gi>=0],k];frame[f'native_embedding_{k}']=values
    frame['native_training_similarity']=np.nan
    write_json(out/'NATIVE_DEV_FEATURE_AUDIT.json',{'source':'frozen train-NTC CONTROL_REFERENCE, not TEST task lookup',
        'feature_effect_labels_used':0,'feature_error_labels_used':0,'mapped_control_gene_task_fraction':float((gi>=0).mean()),
        'mapped_state_fraction':float((si>=0).mean()),'missing_gene_feature_behavior':'official XGBoost native NaN path'})
    write_json(out/'EXPERIMENT_CARD.json',{'trigger':'raw-error fitted trees have many unsplit leaves; compare numerical units on DEV',
        'rows':len(frame),'genes':frame.gene.nunique(),'folds':3,'budget_feedback_genes_each_fold':114,
        'learner':'same official XGBoost 300/depth6/lr0.05/row-col0.8','seed':20260930,
        'features':'Native61 only, no learned Public inputs','methods':['raw_rmse','affine_standardized_raw_rmse'],
        'evaluation_212_used':False,'no_new_algorithm':True,'fit_upper_bound':6,
        'adoption':'DEV paired U20 >=.005; CI lower >=-.005; 60% strata nonnegative; AURC degradation <=5%'})
    parts=[];fit_ledger=[];started=time.monotonic();cpu=time.process_time()
    for fold in range(3):
        train=frame[frame.dev_fold.ne(fold)];genes=ordered_genes(train,2026101001)[:114]
        train=train[train.gene.isin(genes)].reset_index(drop=True);query=frame[frame.dev_fold.eq(fold)].reset_index(drop=True)
        assert not set(train.gene)&set(query.gene)
        xt,xq=native_x(train,query,NATIVE_FEATURES);w=cluster_weights(train);y=train.true_error_rmse.to_numpy(float)
        mu=float(np.average(y,weights=w));sigma=float(np.sqrt(np.average((y-mu)**2,weights=w)))
        if sigma<=1e-12:raise RuntimeError('training labels constant; no unit repair can distinguish ranks')
        part=query[['task_id','gene','target','true_error_rmse']].copy();part['fold']=fold;part['stratum']=part.target.astype(str)+'::fold'+str(fold)
        for method in ['raw_rmse','affine_standardized_raw_rmse']:
            labels=y if method=='raw_rmse' else (y-mu)/sigma
            model=official_gbt(20260930);model.fit(xt,labels,sample_weight=w)
            risk=model.predict(xq);risk=risk if method=='raw_rmse' else risk*sigma+mu
            part[method]=risk
            model.get_booster().save_model(out/f'{method}_fold{fold}.json')
            fit_ledger.append({'method':method,'fold':fold,'training_rows':len(train),'training_gene_clusters':len(genes),
                'label_mean':mu,'label_std':sigma,'transform_fit_record_hash':digest_ids(train.task_id),
                'evaluation_errors_used_for_label_transform':0})
        parts.append(part)
    result=pd.concat(parts,ignore_index=True);result.to_parquet(out/'DEV_OOF_PREDICTIONS.parquet',index=False)
    pd.DataFrame(fit_ledger).to_csv(out/'FIT_AND_TRANSFORM_LEDGER.csv',index=False)
    rows=[]
    for stratum,part in result.groupby('stratum'):
        for method in ['raw_rmse','affine_standardized_raw_rmse']:
            rows.append({'stratum':stratum,'method':method,**point_metrics(part.true_error_rmse,part[method],part.task_id)})
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'STRATA_METRICS.csv',index=False)
    wide=metrics.pivot(index='stratum',columns='method',values='utility');delta=wide.affine_standardized_raw_rmse-wide.raw_rmse
    bootstrap=result.rename(columns={'target':'context','stratum':'target'})
    b=ClusterBootstrap(bootstrap,bootstrap.true_error_rmse)
    stats=summarize_draws(b.difference(result.affine_standardized_raw_rmse,result.raw_rmse),delta.mean())
    aurc=metrics.groupby('method').aurc.mean();aurc_degradation=float(aurc.affine_standardized_raw_rmse/aurc.raw_rmse-1)
    adopt=stats['delta_point']>=.005 and stats['ci95_lower']>=-.005 and (delta>=0).mean()>=.6 and aurc_degradation<=.05
    write_json(out/'DECISION.json',{'status':'COMPLETE','adopt_affine_error_units':bool(adopt),**stats,
        'nonnegative_strata_fraction':float((delta>=0).mean()),'aurc_relative_change':aurc_degradation,
        'fits':6,'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
        'decision_uses_only_542_DEV':True,'next_action':'freeze and run same-budget affine sensitivity' if adopt else 'retain raw RMSE and CDF recipes; no further unit search'})
    print((out/'DECISION.json').read_text(),flush=True)

if __name__=='__main__':main()

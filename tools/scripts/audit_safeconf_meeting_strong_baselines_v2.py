#!/usr/bin/env python3
"""Recompute missing meeting baselines from already released scores, no fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import bootstrap_u20
from tools.scripts.run_safeconf_source_gate_v1 import macro_metric

RUN=Path('/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1')
CACHE=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache')
HOLD=Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/HOLDOUT_FEATURES.parquet')
DOC=ROOT/'docs/组会汇报/20261008_科学示意_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=DOC)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    src_file=RUN/'source_gate_v4/SOURCE_GATE_TASK_PREDICTIONS.csv'
    scores=pd.read_csv(src_file)
    scores=scores[(~scores.shuffle)&(scores.method=='always_source')].copy()
    rows=[];point=[];inputs=[src_file]
    for (source,target),part in scores.groupby(['source','target_model']):
        parts=[]
        for fold in range(5):
            p=CACHE/f'nested_{fold}_{target}_Manual.parquet';inputs.append(p)
            f=pd.read_parquet(p)
            parts.append(f[f.fold==fold][['task_id','target','gene','true_error_rmse','predicted_magnitude']])
        mag=pd.concat(parts,ignore_index=True)
        assert len(mag)==1808 and mag.task_id.is_unique and mag.gene.nunique()==575
        w=part.merge(mag[['task_id','true_error_rmse','predicted_magnitude']],on='task_id',validate='one_to_one',suffixes=('','_check'))
        assert len(w)==1808 and np.allclose(w.true_error_rmse,w.true_error_rmse_check,rtol=1e-12,atol=1e-14)
        for method,col in [('SourceHGB','risk'),('Magnitude','predicted_magnitude')]:
            u,a=macro_metric(w,w[col].to_numpy(float))
            point.append(dict(source=source,target=target,method=method,u20=u,aurc=a,n_tasks=len(w),n_clusters=w.gene.nunique()))
        ci=bootstrap_u20(w,w.risk.to_numpy(float),w.predicted_magnitude.to_numpy(float),5000,20260930)
        rows.append(dict(source=source,target=target,comparison='SourceHGB-minus-Magnitude',**ci))
    pd.DataFrame(point).to_csv(out/'SOURCE_MAGNITUDE_POINT_COMPARISON.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'SOURCE_MAGNITUDE_PAIRED_BOOTSTRAP.csv',index=False)
    hold=pd.read_parquet(HOLD);inputs.append(HOLD)
    system=pd.read_parquet(RUN/'system_freeze_v4/SYSTEM_RANKING.parquet');inputs.append(RUN/'system_freeze_v4/SYSTEM_RANKING.parquet')
    check=hold[['task_id','true_error_rmse']].merge(system[['task_id','true_error_rmse']],on='task_id',validate='one_to_one',suffixes=('_h','_s'))
    assert len(check)==212 and np.allclose(check.true_error_rmse_h,check.true_error_rmse_s,rtol=1e-12,atol=1e-14)
    support=[]
    for name,col in [('NegativeHistorySupport','support_risk'),('PublicRule','simple_history_risk'),('Magnitude','predicted_magnitude')]:
        u,a=macro_metric(hold,hold[col].to_numpy(float))
        support.append(dict(method=name,u20=u,aurc=a,n_tasks=len(hold),n_clusters=hold.gene.nunique(),score_column=col))
    pd.DataFrame(support).to_csv(out/'PUBLIC_SUPPORT_POINT_COMPARISON.csv',index=False)
    audit={'role':'already-seen meeting completeness audit; no model fitting or configuration selection',
           'source_bootstrap_replicates':5000,'bootstrap_unit':'gene cluster, shared across target contexts',
           'truth_contract_exact_match':True,'new_model_fits':0,'new_download_bytes':0,'new_gpu_hours':0,
           'permanent_test_truth_opened':False,'wall_seconds':time.monotonic()-started,
           'source_scores':'frozen v4 train-OOF channel ranks',
           'magnitude_scores':'raw predicted magnitude from the exact target-model fold cache',
           'support_score':'pre-existing support_risk; no sign selection on evaluation results',
           'inputs':{str(p):sha(p) for p in inputs}}
    (out/'STRONG_BASELINE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(point).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(support).to_string(index=False))


if __name__=='__main__':main()

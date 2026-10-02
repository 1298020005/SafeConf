#!/usr/bin/env python3
"""Fixed stronger Public rules on DEV; no fits, no test arrays, no label selection."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.scripts.run_safeconf_publicset_v1 import (
    OUT,RUNTIME,load_domain,score_records,summarize,atomic_json,atomic_csv,file_hash,
)


def main():
    out=OUT/'strong_simple_reference_v1'; runtime=RUNTIME/'strong_simple_reference_v1'
    if (out/'RUN_STATUS.json').exists(): raise FileExistsError('completed rule version immutable')
    out.mkdir(parents=True,exist_ok=True); runtime.mkdir(parents=True,exist_ok=True)
    registration={'role':'DEV/SEEN fixed strong-simple comparator','rules':{
        'NearestControl':'minimum prediction-time control RMSE; ties by experiment_id',
        'SameContextSupport':'support mean among same-context histories if any; otherwise all eligible histories',
        'SameConditionSupport':'support mean among same-condition histories if any; otherwise all eligible histories'},
        'hyperparameters':None,'new_fits':0,'new_upstream_calls':0,'MC_TEST_reads':0,
        'scope':'same allowed query/history tables as full PublicSet; no outcome-selected subset',
        'registered_before_metric_computation':True,'code_sha256':file_hash(__file__)}
    atomic_json(out/'CONFIG.json',registration); started=time.monotonic(); records=[]
    for name in ['McFaline','Source']:
        domain=load_domain(name)
        for fold in range(5):
            rows=np.flatnonzero(domain.tasks.fold.to_numpy()==fold)
            groups=domain.groups(rows,domain.forbidden(rows))
            weights={key:{} for key in registration['rules']}
            for g in groups:
                if not len(g['ix']): continue
                task=domain.tasks.iloc[g['q']]; m=domain.memory.iloc[g['ix']]
                rmse=g['x'][:,domain.fields.index('control_rmse')]
                winner=np.lexsort((m.experiment_id.astype(str).to_numpy(),rmse))[0]
                w=np.zeros(len(rmse));w[winner]=1.;weights['NearestControl'][g['q']]=w
                for key,flag in [('SameContextSupport',m.context.astype(str).eq(str(task.context)).to_numpy()),
                                 ('SameConditionSupport',m.condition.astype(str).eq(str(task.condition)).to_numpy())]:
                    w=g['s']*flag if flag.any() else g['s'].copy();w=w/w.sum();weights[key][g['q']]=w
            for builder,w in weights.items():
                values,ids,priors,ws=score_records(domain,groups,w,builder,0,fold)
                records.extend(values)
                destination=runtime/f'{name}/outer{fold}';destination.mkdir(parents=True,exist_ok=True)
                np.savez(destination/f'{builder}_PRIORS.npz',task_ids=np.asarray(ids),priors=priors)
                atomic_csv(destination/f'{builder}_WEIGHTS.csv',pd.DataFrame(ws))
    frame,contexts,macro=summarize(records)
    atomic_csv(out/'TASK_PREDICTIONS.csv.gz',frame);atomic_csv(out/'CONTEXT_RESULTS.csv',contexts);atomic_csv(out/'MACRO_RESULTS.csv',macro)
    atomic_json(out/'RUN_STATUS.json',registration|{'status':'COMPLETE','elapsed_seconds':time.monotonic()-started,'n_prediction_rows':len(frame),
        'results_sha256':file_hash(out/'TASK_PREDICTIONS.csv.gz')})
    print(macro.groupby(['domain','upstream','builder']).utility20.mean().to_string())


if __name__=='__main__':main()

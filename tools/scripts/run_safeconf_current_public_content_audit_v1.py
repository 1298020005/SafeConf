#!/usr/bin/env python3
"""Current 212-task PublicRule content audit, no model fitting or new truth.

Keep exact-state exclusion and same-context preference. Reuse the five fixed
support-quartile donor orders; exclude current query units before permutation.
"""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.safeconf_continual.research import bootstrap_u20,metrics,ids_hash
from tools.scripts.run_safeconf_public_mechanisms import permutation

COMMON=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis')
HOLD=Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1/HOLDOUT_FEATURES.parquet')
OUT=Path('/home/yyf/runtime_artifacts/safeconf_goal_report_20261008_v1/current_public_content')


def macro(f,s):
    values=[]
    for _,q in f.assign(_score=s).groupby('target'):
        values.append(metrics(q,q._score.to_numpy())['utility20'])
    return float(np.nanmean(values))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'STATUS.json').exists():raise FileExistsError('immutable version')
    start=time.monotonic();cpu=time.process_time()
    f=pd.read_parquet(HOLD).reset_index(drop=True)
    assert len(f)==212 and f.gene.nunique()==152
    tasks=pd.read_csv(COMMON/'TEST_TASKS.csv').set_index('task_id')
    ix=tasks.reset_index().reset_index().set_index('task_id').loc[f.task_id]['index'].to_numpy(int)
    pred=np.asarray(np.load(COMMON/'TEST_CALIBRATED_EFFECTS.npy',mmap_mode='r')[ix],float)
    mem=pd.read_parquet(COMMON/'public_mcfaline_trainval/public_memory.parquet').sort_values('effect_vector_row').reset_index(drop=True)
    effects=np.asarray(np.load(COMMON/'reference_estimand_diagnostic/CELL_WEIGHTED_PUBLIC_EFFECTS.npy',mmap_mode='r'),float)
    assert effects.shape[1]==pred.shape[1]==2840 and np.array_equal(mem.effect_vector_row,np.arange(len(mem)))
    forbidden=set('McFaline23::'+f.task_id.astype(str));allowed=~mem.experiment_id.astype(str).isin(forbidden).to_numpy()
    gene_rows=mem.groupby('perturbation_target',sort=False).indices;groups=[]
    for q,row in enumerate(f.itertuples(index=False)):
        ids=np.asarray(gene_rows.get(str(row.gene),[]),int);ids=ids[allowed[ids]]
        mm=mem.iloc[ids];ids=ids[~((mm.context.astype(str)==str(row.context))&(mm.condition.astype(str)==str(row.treatment))).to_numpy()]
        mm=mem.iloc[ids];same=mm.context.astype(str).to_numpy()==str(row.context)
        chosen=ids[same] if same.any() else ids
        assert len(chosen)>0
        counts=mem.iloc[chosen].n_cells.to_numpy(float);w=counts/counts.sum()
        groups.append((q,chosen,w))
    def score(donor):
        return np.array([np.sqrt(w@np.mean((effects[donor[ids]]-pred[q])**2,axis=1)) for q,ids,w in groups])
    real=score(np.arange(len(mem)))
    assert np.allclose(real,f.simple_history_risk,rtol=1e-5,atol=1e-8)
    k=43
    assert np.array_equal(np.lexsort((f.task_id.astype(str),-real))[:k],np.lexsort((f.task_id.astype(str),-f.simple_history_risk))[:k])
    # Primary true score remains byte-for-byte the saved current rule.
    real=f.simple_history_risk.to_numpy(float)
    config={'role':'SEEN current-default mechanism audit; not parameter selection','n_tasks':212,'n_genes':152,
        'policy':'exclude all current query experimental IDs and exact state; prefer same context',
        'permutation':'same study/axis/context/condition/support quartile; unchanged recipient counts/weights',
        'seeds':[20260930+i for i in range(5)],'fits':0,'new_gpu_hours':0,'new_download_bytes':0,
        'current_query_units_in_bank':int((~allowed).sum()),'current_true_score_reproduced':True,
        'scoring_truth':'already opened HOLDOUT_FEATURES error only; no raw test truth arrays',
        'method_changed':False,'support_matching_is_quartile_not_exact_count':True}
    (OUT/'CONFIG.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    rows=[];audits=[];per=f[['task_id','target','gene','true_error_rmse']].copy();per['PublicRule']=real
    base=macro(f,real)
    rows.append({'method':'PublicRule','u20':base})
    allowed_ids=np.flatnonzero(allowed)
    for order,seed in enumerate(config['seeds']):
        local,record=permutation(mem.iloc[allowed_ids].reset_index(drop=True),seed,'McFaline')
        donor=np.arange(len(mem));donor[allowed_ids]=allowed_ids[local]
        null=score(donor);name=f'CurrentContentNull_{order}';per[name]=null
        result=bootstrap_u20(f,real,null,5000,20260930)
        rows.append({'method':name,'u20':macro(f,null),**result})
        used=np.concatenate([ids for _,ids,_ in groups])
        audits.append({'order':order,'seed':seed,'used_history_rows':len(used),'moved_fraction':float(np.mean(donor[used]!=used)),
            'different_gene_fraction':float(np.mean(mem.perturbation_target.to_numpy()[donor[used]]!=mem.perturbation_target.to_numpy()[used])),
            'forbidden_donors_used':int(np.sum(~allowed[donor[used]])),'groups':len(record)})
        assert audits[-1]['forbidden_donors_used']==0
    for name,col in [('FrozenHistorySupport','support_risk'),('Magnitude','predicted_magnitude')]:
        result=bootstrap_u20(f,real,f[col].to_numpy(float),5000,20260930)
        rows.append({'method':name,'u20':macro(f,f[col].to_numpy(float)),**result})
    per.to_parquet(OUT/'PER_TASK_SCORES.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'RESULTS.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'PERMUTATION_AUDIT.csv',index=False)
    status=config|{'status':'COMPLETE','wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
        'task_hash':ids_hash(f.task_id),'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_holdout_sha256':hashlib.sha256(HOLD.read_bytes()).hexdigest()}
    (OUT/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(audits).to_string(index=False))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Audit the locked Tahoe pretruth extraction without touching sealed treated test cells."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pyarrow.parquet as pq

DEFAULT_EXPR=Path('/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M/safeconf_e263_pretruth/E263_SELECTED_PRETRUTH.parquet')
DEFAULT_PANEL=Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/E260_tahoe_metadata_panel_20260925/LOCKED_PANEL.csv')

def main(a):
    out=a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True)
    panel=pd.read_csv(a.panel,dtype=str)
    panel['maximum_cells_to_extract']=pd.to_numeric(panel['maximum_cells_to_extract'])
    pkeys=set(tuple(x) for x in panel.loc[panel.split.isin(['train','validation']),['cell_line','drug_dose','plate','split']].itertuples(index=False,name=None))
    # Metadata pass is intentionally separate, so a malformed key cannot be hidden by expression parsing.
    pf=pq.ParquetFile(a.expression)
    counts=defaultdict(int); roles=defaultdict(int); bad=[]; n=0; n_test_treated=0
    t0=time.time()
    for batch in pf.iter_batches(batch_size=4096,columns=['cell_line','drug','plate','drug_dose','split','role']):
        d=batch.to_pydict(); m=len(d['split']); n+=m
        for i in range(m):
            key=(str(d['cell_line'][i]),str(d['drug_dose'][i]),str(d['plate'][i]),str(d['split'][i]))
            roles[str(d['role'][i])]+=1; counts[key]+=1
            if str(d['role'][i])=='test_treated': n_test_treated+=1
    rows=[]
    for key,ncells in sorted(counts.items()):
        cl,dd,plate,split=key; rows.append({'cell_line':cl,'drug_dose':dd,'plate':plate,'split':split,'n_selected_cells':ncells,
          'panel_match':key in pkeys,'panel_cap':int(panel.loc[(panel.cell_line==cl)&(panel.drug_dose==dd)&(panel.plate==plate)&(panel.split==split),'maximum_cells_to_extract'].iloc[0]) if key in pkeys else None})
    table=pd.DataFrame(rows); table.to_csv(out/'TASK_COUNTS.csv',index=False)
    # Expression pass: summary only; never writes test treated data.
    expr_rows=0; nonzero_sum=0; nonzero_min=10**9; nonzero_max=0; finite_bad=0; value_min=math.inf; value_max=-math.inf; value_sum=0.; value_sq=0.; value_n=0
    by_role=defaultdict(lambda:[0,0,0,0.0,0.0])
    for batch in pf.iter_batches(batch_size=512,columns=['split','role','genes','expressions']):
        d=batch.to_pydict(); m=len(d['split']); expr_rows+=m
        for i in range(m):
            genes=d['genes'][i] or []; vals=d['expressions'][i] or []; ll=len(genes); nonzero_sum+=ll; nonzero_min=min(nonzero_min,ll); nonzero_max=max(nonzero_max,ll)
            arr=np.asarray(vals,dtype=float); finite=np.isfinite(arr); finite_bad+=int((~finite).sum())
            if finite.any():
                vv=arr[finite]; value_min=min(value_min,float(vv.min())); value_max=max(value_max,float(vv.max())); value_sum+=float(vv.sum()); value_sq+=float(np.dot(vv,vv)); value_n+=int(vv.size)
            r=str(d['role'][i]); by_role[r][0]+=1; by_role[r][1]+=ll; by_role[r][2]+=int(finite.sum()); by_role[r][3]+=float(arr[finite].sum()) if finite.any() else 0.; by_role[r][4]+=int((~finite).sum())
    role_rows=[]
    for r,(nr,ng,nf,sv,nb) in sorted(by_role.items()): role_rows.append({'role':r,'n_cells':nr,'mean_n_genes':ng/max(1,nr),'n_finite_values':nf,'n_nonfinite_values':nb,'mean_expression_over_finite':sv/max(1,nf)})
    pd.DataFrame(role_rows).to_csv(out/'EXPRESSION_ROLE_SUMMARY.csv',index=False)
    status={'status':'TAHOE_RAW_HISTORY_AUDIT_COMPLETE','expression_file':str(a.expression),'panel_file':str(a.panel),
      'parquet_rows':pf.metadata.num_rows,'metadata_rows_scanned':n,'expression_rows_scanned':expr_rows,'role_counts':dict(roles),
      'n_test_treated_records_seen':n_test_treated,'n_train_validation_panel_keys':len(pkeys),'n_task_keys_in_extract':len(counts),
      'all_train_validation_keys_match_panel':bool(table.loc[table.split.isin(['train','validation']),'panel_match'].all()),
      'max_selected_cells_per_key':int(table.n_selected_cells.max()),'min_genes_per_cell':int(nonzero_min),'max_genes_per_cell':int(nonzero_max),
      'finite_value_fraction':float(value_n/max(1,value_n+finite_bad)),'value_min':value_min,'value_max':value_max,
      'scope':'Provenance/quality audit only. No test treated expression is aggregated or evaluated.'}
    (out/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--expression',type=Path,default=DEFAULT_EXPR); p.add_argument('--panel',type=Path,default=DEFAULT_PANEL); p.add_argument('--output-dir',type=Path,required=True); main(p.parse_args())

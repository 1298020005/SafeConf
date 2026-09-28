#!/usr/bin/env python3
"""Summarize the completed fixed batch; never fit a model or select new trials."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/实验结果/ModelDecision_light_batch_20260928'
s=json.loads((OUT/'RUN_STATUS.json').read_text())
assert s['status']=='COMPLETE_STOPPED_AFTER_FIRST_BATCH'
r=pd.read_csv(OUT/'RESULTS.csv')
d=pd.read_csv(OUT/'DATASET_RESULTS.csv')
m=pd.read_csv(OUT/'MACRO.csv').set_index('method')
c=pd.read_csv(OUT/'PAIRED_COMPARISONS.csv')
# Audit aggregation and task identity, including identical folds across predictors.
z=pd.read_csv(OUT/'SPLIT_MANIFEST.csv')
for (dataset,pred,fold),g in z.groupby(['dataset','predictor','fold']):
    a=g[g.role=='fit'];b=g[g.role=='eval']
    assert not set(a.task_key)&set(b.task_key)
    assert not set(a.perturbation)&set(b.perturbation)
for (dataset,task),g in z[z.role=='eval'].groupby(['dataset','task_key']):
    assert len(g)==3 and g['fold'].nunique()==1
assert z[z.role=='eval'][['dataset','task_key']].drop_duplicates().shape[0]==552
assert r.groupby(['dataset','predictor','fold']).method.nunique().eq(17).all()

names=['Cui','Frangieh','Lara ex vivo','Lara in vivo','McFarland','Santinha','sciPlex3']
order=sorted(d.dataset.unique())
fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1,1.45]})
x=np.arange(4)
for kind,color in [('Ridge','#31688e'),('HGB','#35b779'),('MLP','#db744d')]:
    axes[0].plot(x,[m.loc[kind+'_'+g,'utility20'] for g in ['M','P','PQ','PQH']],marker='o',label=kind,color=color,lw=2)
axes[0].axhline(m.loc['Magnitude_raw','utility20'],color='#555555',ls='--',label='Raw magnitude')
axes[0].set_xticks(x,['M','P','P + Q','P + Q + H'])
axes[0].set_ylabel('Top-20% review utility (U20)');axes[0].legend(frameon=False)
axes[0].set_title('Fixed-budget development comparison')
wide=d.pivot(index='dataset',columns='method',values='utility20').loc[order]
vals=np.array([(wide[k+'_PQH']-wide[k+'_PQ']).to_numpy() for k in ['Ridge','HGB','MLP']]).T
v=max(.02,float(np.abs(vals).max()))
im=axes[1].imshow(vals,cmap='RdBu',vmin=-v,vmax=v,aspect='auto')
axes[1].set_xticks(range(3),['Ridge','HGB','MLP']);axes[1].set_yticks(range(7),names)
for (i,j),val in np.ndenumerate(vals): axes[1].text(j,i,f'{val:+.3f}',ha='center',va='center',color='white' if abs(val)>v*.58 else 'black')
axes[1].set_title('History content increment after P + Q')
fig.colorbar(im,ax=axes[1],label='Change in U20',shrink=.85)
fig.suptitle('552 validation tasks; 3 related upstream baselines; development only',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'MODEL_COMPARISON.png',dpi=180);fig.savefig(OUT/'MODEL_COMPARISON.pdf');plt.close(fig)
# A compact, stable table for the report: dataset equality; types shown separately.
domain=pd.read_csv(OUT/'DOMAIN_MACRO.csv').set_index(['type','method'])
rows=[]
for name in ['Magnitude_raw','Magnitude_anchor']+[k+'_'+g for k in ['Ridge','HGB','MLP'] for g in ['M','P','PQ','PQH','anchored_PQH']]:
    rows.append({'方法':name,'总体 U20':m.loc[name,'utility20'],'基因 U20':domain.loc[('gene',name),'utility20'],'化学 U20':domain.loc[('chemical',name),'utility20'],'总体 Spearman':m.loc[name,'spearman']})
t=pd.DataFrame(rows)
(OUT/'RESULT_TABLE.md').write_text(t.to_markdown(index=False,floatfmt='.4f')+'\n')
(OUT/'COMPARISON_TABLE.md').write_text(c.to_markdown(index=False,floatfmt='.4f')+'\n')
summary={'summary_checks_passed':True,'n_outer_cells':63,'n_rows':len(r),'n_methods':17,'n_unique_tasks':552,'main_comparison_fits':945,'auxiliary_magnitude_fits_including_inner_crossfit':252,'same_splits_across_predictors':True,'no_outer_task_or_perturbation_overlap':True,'aggregation':'fold -> predictor -> dataset-group equal weighting; gene and chemical reported separately','intervals':'not estimated: pilot, related groups, and tiny folds; fold count is not independent n','no_additional_model_fits':True}
(OUT/'SUMMARY_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(t.to_string(index=False));print(c.to_string(index=False))

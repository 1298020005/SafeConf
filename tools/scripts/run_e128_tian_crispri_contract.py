#!/usr/bin/env python3
"""E128: freeze a four-batch Tian/Kampmann 2021 CRISPRi rectangle."""
from __future__ import annotations
import hashlib,json,math
from datetime import datetime
from pathlib import Path
import anndata as ad,pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/TianKampmann2021_CRISPRi.h5ad')
VOCAB=Path('/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human/vocab.json')
OUT=ROOT/'docs/实验结果/E128_tian_crispri_four_batch_contract_20260714'
MIN_CELLS=30;FRACTIONS=(.25,.5,.75,1.0);INVALID={'','control','ctrl','nan','none','noise','nt1'}
def key(*x):return hashlib.sha256('|'.join(map(str,x)).encode()).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return 'sha256:'+h.hexdigest()
def main():
 for d in (OUT/'manifests',OUT/'tables',OUT/'reports'):d.mkdir(parents=True,exist_ok=True)
 a=ad.read_h5ad(SOURCE,backed='r');obs=a.obs[['batch','perturbation']].astype(str).copy();shape=tuple(map(int,a.shape));gene_axis=set(a.var_names.astype(str));a.file.close();vocab=json.loads(VOCAB.read_text())
 counts=obs.groupby(['batch','perturbation']).size().rename('n_cells').reset_index();mat=counts.pivot(index='batch',columns='perturbation',values='n_cells').fillna(0).astype(int);contexts=sorted(mat.index.astype(str));shared=mat.columns[mat.ge(MIN_CELLS).all(0)].astype(str).tolist()
 candidates=sorted(p for p in shared if p.lower() not in INVALID);excluded_gene=[p for p in candidates if p not in gene_axis];excluded_vocab=[p for p in candidates if p in gene_axis and p.upper() not in vocab];perts=[p for p in candidates if p in gene_axis and p.upper() in vocab]
 if 'control' not in shared or len(contexts)!=4 or len(perts)<80:raise RuntimeError(f'unexpected rectangle {len(contexts)}x{len(perts)}')
 count_map={(r.batch,r.perturbation):int(r.n_cells) for r in counts.itertuples(index=False)};rows=[]
 for fi,held in enumerate(contexts,1):
  fold=f'Tian_CRISPRi_batch_holdout_{fi}_B{held}';sources=[c for c in contexts if c!=held];nnew=max(10,int(math.ceil(.2*len(perts))));ordered=sorted(perts,key=lambda p:key('E128',fold,'pert',p));new,seen=set(ordered[:nnew]),set(ordered[nnew:]);pairs=sorted([(c,p) for c in sources for p in sorted(seen)],key=lambda x:key('E128',fold,'pair',*x));naux=min(30,max(20,int(round(.1*len(pairs)))));val,random=set(pairs[:naux]),set(pairs[naux:2*naux]);base=[x for x in pairs if x not in val and x not in random];membership={}
  for c in sources:
   cp=sorted([x for x in base if x[0]==c],key=lambda x:key('E128',fold,'fraction',*x))
   for f in FRACTIONS:
    selected=set(cp[:max(1,int(round(len(cp)*f)))])
    for pair in cp:membership.setdefault(pair,{})[f]=pair in selected
  for c in contexts:
   for p in perts:
    pair=(c,p)
    if c==held and p in new:split,setting='test','context_and_perturbation_unseen'
    elif c==held:split,setting='test','context_unseen_row'
    elif p in new:split,setting='test','perturbation_unseen_column'
    elif pair in val:split,setting='val','source_validation_pair'
    elif pair in random:split,setting='test','random_missing_pair'
    else:split,setting='train','source_train_pair'
    row={'dataset':'Tian_CRISPRi','modality':'gene_CRISPRi_technical_batch_shift','fold_id':fold,'heldout_context':held,'source_contexts':'+'.join(sources),'split':split,'setting':setting,'context':c,'batch':c,'perturbation':p,'n_cells':count_map[pair],'perturbation_seen_in_training':p in seen,'context_seen_in_training':c in sources,'selected_without_expression_values':True}
    for f in FRACTIONS:row[f'in_train_fraction_{int(f*100)}']=bool(split=='train' and membership.get(pair,{}).get(f,False))
    rows.append(row)
 manifest=pd.DataFrame(rows).sort_values(['fold_id','split','setting','context','perturbation'])
 for fold,g in manifest.groupby('fold_id'):
  if len(g)!=len(contexts)*len(perts) or g.duplicated(['context','perturbation']).any():raise RuntimeError(f'{fold}: partition invariant')
  previous=set()
  for f in FRACTIONS:
   selected=set(map(tuple,g.loc[g[f'in_train_fraction_{int(f*100)}'],['context','perturbation']].to_numpy()))
   if not previous.issubset(selected):raise RuntimeError(f'{fold}: fractions not nested')
   previous=selected
 manifest.to_csv(OUT/'manifests/E128_TASK_MANIFEST.csv',index=False);summary=manifest.groupby(['fold_id','setting','split'],as_index=False).agg(n_tasks=('perturbation','size'),min_cells=('n_cells','min'),median_cells=('n_cells','median'));summary.to_csv(OUT/'tables/E128_SETTING_SUMMARY.csv',index=False);pd.DataFrame({'context':contexts,'meaning':['technical replicate batch']*len(contexts)}).to_csv(OUT/'tables/E128_CONTEXTS.csv',index=False);pd.DataFrame({'perturbation':perts}).to_csv(OUT/'tables/E128_PERTURBATIONS.csv',index=False)
 status={'experiment':'E128_tian_crispri_four_batch_contract','generated_at':datetime.now().isoformat(timespec='seconds'),'status':'contract_frozen','source':str(SOURCE),'source_sha256':sha(SOURCE),'source_shape':shape,'context_definition':'technical sequencing/experimental batch, not a biological cell type','n_contexts':len(contexts),'n_perturbations':len(perts),'excluded_targets_missing_gene_axis':excluded_gene,'excluded_targets_missing_scgpt_vocab':excluded_vocab,'n_folds':int(manifest.fold_id.nunique()),'n_manifest_rows':len(manifest),'minimum_cells_per_pair':int(manifest.n_cells.min()),'expression_values_read_during_selection':False,'gene_identity_and_vocab_read_for_predictor_compatibility':True,'target_effect_prediction_or_error_used_for_selection':False};(OUT/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
 (OUT/'reports/E128_CONTRACT_REPORT.md').write_text(f"# E128｜Tian–Kampmann CRISPRi 四批次冻结合同\n\n四个 context 是技术批次，不解释成不同细胞类型。合同只用标签、每格细胞数、基因身份轴和 scGPT 词表冻结 {len(contexts)} × {len(perts)} 个矩阵；每格至少 {manifest.n_cells.min()} 个细胞。表达值、效应、预测和误差未用于选择。每折包含随机 pair、整批次、整扰动列和双未见。\n")
 (OUT/'README_先看这个.md').write_text('# E128 先看这个\n\n先读 `reports/E128_CONTRACT_REPORT.md`。\n');print(json.dumps(status,ensure_ascii=False,indent=2));print(summary.to_string(index=False))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Read-only PerturbMap qualification audit; no expression or permanent TEST truth."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/perturbmap_audit_v1'
PUBLIC=Path('/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201/public_memory.parquet')
ELIG=Path('/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201/eligibility.parquet')
GENES=Path('/home/yyf/data/safeconf_dual_memory_20260929/public_txpert_e201/gene_ids.json')
TASKS=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/common_gene_axis/risk_cache/TX_TASK_SPLIT.csv')
ORION_META=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_allowed_biology_20261002_v2/QUERY_METADATA_ONLY.csv')
ORION_MANIFEST=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_allowed_biology_20261002_v2/BIOLOGY_MANIFEST.json')
ORION_AXIS=Path('/home/yyf/runtime_artifacts/safeconf_research_20261001/orion_allowed_biology_20261002_v2/ENDPOINT_GENE_MANIFEST.csv')
SAMS_STATUS=Path('/home/yyf/runtime_artifacts/safeconf_research_20261003/sams_v1/TRAINING_STATUS.json')

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def bind(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 m=pd.read_parquet(PUBLIC); e=pd.read_parquet(ELIG); t=pd.read_csv(TASKS); o=pd.read_csv(ORION_META); axis=pd.read_csv(ORION_AXIS)
 genes=json.loads(GENES.read_text())['gene_ids']; mgenes=set(m.perturbation_target.astype(str)); ogenes=set(axis.gene_name.astype(str))
 routes=[]
 for source in sorted(m.context.astype(str).unique()):
  sg=set(m.loc[m.context.eq(source),'perturbation_target'].astype(str))
  for target in sorted(o.context.astype(str).unique()):
   target_trainval=set(o.loc[(o.context.eq(target))&o.role.isin(['TRAIN','VALIDATION']),'gene'].astype(str))
   routes.append({'source_context':source,'target_context':target,'source_records':int(m.context.eq(source).sum()),'source_unique_targets':len(sg),'target_train_validation_units':int(((o.context==target)&o.role.isin(['TRAIN','VALIDATION'])).sum()),'target_train_validation_unique_targets':len(target_trainval),'paired_target_identities':len(sg&target_trainval),'source_target_gene_metadata_overlap':len(sg&ogenes),'status':'QUALIFIED_METADATA_ONLY' if len(sg&target_trainval)>=60 else 'INSUFFICIENT_PAIRS'})
 pd.DataFrame(routes).to_csv(OUT/'PAIR_COVERAGE.csv',index=False)
 # Natural E201 queries: exact same-context public history availability.
 per=[]
 for r in t.itertuples():
  h=m[m.perturbation_target.eq(str(r.gene))]
  same=int(h.context.eq(str(r.target)).sum());cross=int((~h.context.eq(str(r.target))).sum())
  per.append({'task_id':r.task_id,'gene':r.gene,'target_context':r.target,'same_context_history':same,'cross_context_history':cross,'natural_cross_only':same==0 and cross>0,'no_history':same==0 and cross==0})
 q=pd.DataFrame(per);q.to_csv(OUT/'E201_QUERY_HISTORY_COVERAGE.csv',index=False)
 # Information budget and contract binding; no target expression opened by this audit.
 info=[
  {'information':'E201_public_history','records':len(m),'unique_targets':m.perturbation_target.nunique(),'contexts':m.context.nunique(),'role':'source/query history','used_by_audit':True},
  {'information':'E201_eligibility_relation','records':len(e),'unique_target_contexts':e.target_context.nunique(),'role':'eligibility only','used_by_audit':True},
  {'information':'E201_source_tasks','records':len(t),'natural_cross_only_tasks':int(q.natural_cross_only.sum()),'no_history_tasks':int(q.no_history.sum()),'role':'query metadata only','used_by_audit':True},
  {'information':'Orion_train_validation_metadata','records':len(o[o.role.isin(['TRAIN','VALIDATION'])]),'unique_targets':o.loc[o.role.isin(['TRAIN','VALIDATION']),'gene'].nunique(),'contexts':o.loc[o.role.isin(['TRAIN','VALIDATION']),'context'].nunique(),'role':'candidate target-background anchors; not read as mapping outputs','used_by_audit':True},
  {'information':'Orion_permanent_test_truth','records':0,'role':'protected','used_by_audit':False},
  {'information':'source_error_labels','records':0,'role':'not used by qualification audit','used_by_audit':False},
  {'information':'target_error_labels','records':0,'role':'not used by qualification audit','used_by_audit':False},
 ]
 pd.DataFrame(info).to_csv(OUT/'INFORMATION_BUDGET.csv',index=False)
 sams=json.loads(SAMS_STATUS.read_text()) if SAMS_STATUS.exists() else {}
 upstream=[
 {'upstream_model_id':'TxPert_GAT','model_version':'e201_official_frozen','architecture_family':'TxPert','qualification':'existing_registered_source; source errors available','prediction_contract':'E201_common2840gene_log1p_delta_v1','status':'QUALIFIED_FOR_EXISTING_SOURCE_AUDITS'},
 {'upstream_model_id':'TxPert_Exphormer','model_version':'e201_official_frozen','architecture_family':'TxPert','qualification':'existing_registered_source; source errors available','prediction_contract':'E201_common2840gene_log1p_delta_v1','status':'QUALIFIED_FOR_EXISTING_SOURCE_AUDITS'},
 {'upstream_model_id':'SAMS_VAE','model_version':str(sams.get('best_checkpoint')),'architecture_family':'SAMS-VAE','qualification':'pending true metadata-only generation','prediction_contract':'McFaline generative counterfactual','status':str(sams.get('upstream_competence_status','PENDING_TRUE_GENERATION'))},
 ]
 pd.DataFrame(upstream).to_csv(OUT/'UPSTREAM_QUALIFICATION.csv',index=False)
 manifest=json.loads(ORION_MANIFEST.read_text())
 receipt={'schema':'safeconf_perturbmap_qualification_v1','status':'QUALIFICATION_COMPLETE_NO_MODEL_FIT','decision':'REVISE_BEFORE_EXPERIMENT','public_history':bind(PUBLIC),'eligibility':bind(ELIG),'e201_tasks':bind(TASKS),'orion_metadata':bind(ORION_META),'orion_manifest':bind(ORION_MANIFEST),'orion_axis':bind(ORION_AXIS),'e201_tasks_total':len(t),'e201_same_context_tasks':int((q.same_context_history>0).sum()),'e201_natural_cross_only_tasks':int(q.natural_cross_only.sum()),'e201_no_history_tasks':int(q.no_history.sum()),'orion_roles':manifest.get('roles'),'orion_contexts':manifest.get('contexts'),'orion_train_validation_units':int(o.role.isin(['TRAIN','VALIDATION']).sum()),'orion_target_anchor_truth_opened_by_audit':False,'permanent_test_truth_opened_by_audit':False,'public_target_identity_overlap_E201_vs_Orion_endpoint3285':len(mgenes&ogenes),'gene_axis_overlap_E201_3352_vs_Orion_endpoint3285':len(set(genes)&ogenes),'e201_public_gene_axis':len(genes),'orion_endpoint_gene_axis':len(ogenes),'recommendation':'Do not run PerturbMap on E201 as natural missing-background claim; if pursued, use Orion TRAIN/VALIDATION anchors in an isolated retrospective DEV contract and report it as background-transfer development.'}
 (OUT/'QUALIFICATION_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

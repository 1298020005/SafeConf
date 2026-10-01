"""Persist metadata-only inventory and qualification of existing assets."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
REPO=Path('/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921')
OLD=Path('/home/yyf/proj/docs/实验结果')
DOC=REPO/'docs/实验结果'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

panelnames=['E60_gears_fixed_panel_formal_20260711','E66_norman_gears_fixed_panel_formal_20260711','E71_frangieh_gears_fixed_panel_formal_20260711','E75a_adamson_gears_panel2_20260711','E75b_norman_gears_panel2_20260711','E75c_frangieh_gears_panel2_20260711','E78a_adamson_gears_panel3_20260711']
panels=[];pairs=[];genes=[]
for exp in panelnames:
 s=json.loads((OLD/exp/'raw_gears/seed_11/GEARS_PREDICTION_RECORD_STATUS.json').read_text())
 for c in s['fixed_test_conditions']:pairs.append((s['dataset'],c));genes.append(c.replace('+ctrl',''))
 for seed in [11,22,33]:
  status=OLD/exp/f'raw_gears/seed_{seed}/GEARS_PREDICTION_RECORD_STATUS.json'
  if not status.is_file():
   panels.append({'experiment':exp,'dataset':s['dataset'],'seed':seed,'n_tasks':0,'prediction_exists':False,'checkpoint_exists':False,'status_path':str(status),'status_missing':True})
   continue
  s1=json.loads(status.read_text());p=Path(s1['predicted_npz']);shape=None
  with zipfile.ZipFile(p) as z:
   with z.open(z.namelist()[0]) as f:
    v=np.lib.format.read_magic(f)
    reader=np.lib.format.read_array_header_1_0 if v==(1,0) else np.lib.format.read_array_header_2_0
    shape=reader(f)[0]
  panels.append({'experiment':exp,'dataset':s1['dataset'],'seed':seed,'n_tasks':s1['n_prediction_records'],'n_native_genes':int(shape[0]),'prediction_path':str(p),'prediction_exists':p.is_file(),'checkpoint_path':str(p.parent.parent/'model/model.pt'),'checkpoint_exists':(p.parent.parent/'model/model.pt').is_file(),'registered_old_test_truth':True,'truth_file_opened_for_this_audit':False,'status_path':str(status)})
pd.DataFrame(panels).to_csv(OUT/'ASSET_AGENT_AUDIT.old_gears_panels.csv',index=False)

seal=json.loads((DOC/'E201_txpert_multitarget_retraining_20260802/E201_FAMILY_SEAL.json').read_text())
tx=[]
for row in seal['records']:
 path=Path(str(row['last']['path']).replace('DATA/','/home/yyf/data/',1))
 run=path.parent.parent
 status=json.loads((run/'E201_RUN_STATUS.json').read_text())
 metrics=run/'logs/txpert/version_0/metrics.csv'
 m=pd.read_csv(metrics).dropna(subset=['val_pearson_delta'])
 h=sha(path)
 tx.append({'target':row['target'],'seed':row['seed'],'checkpoint_path':str(path),'checkpoint_sha256':h,'checkpoint_hash_match':h==row['last']['sha256'],'native_genes':3352,'n_source_train_rows':status['n_train_dataset_rows'],'n_source_validation_rows':status['n_validation_dataset_rows'],'validation_log_path':str(metrics),'last_logged_source_val_pearson_delta':float(m.iloc[-1].val_pearson_delta),'max_logged_source_val_pearson_delta':float(m.val_pearson_delta.max()),'sealed_best_checkpoint_score':row['best_source_validation_score'],'training_blind_view_path':f"/home/yyf/data/txpert_official_20260802/cache/E201_blind_{row['target']}/de_adata_test.h5ad",'condition_split_path':f"/home/yyf/data/txpert_official_20260802/cache/E201_blind_{row['target']}/splits/train_test_split.pkl",'new_training':False})
assert all(x['checkpoint_hash_match'] for x in tx)
pd.DataFrame(tx).to_csv(OUT/'ASSET_AGENT_AUDIT.txpert_validation_inventory.csv',index=False)

candidates=[
 {'asset':'E195_native_GEARS_UQ_Norman','native_genes':5025,'persisted_checkpoints':6,'registered_evaluation_tasks':48,'distinct_evaluation_gene_clusters':48,'registered_validation_conditions_per_panel':20,'distinct_validation_conditions_union':24,'expression_download_bytes':0,'new_training_gpu_hours':0,'can_run_within_one_day':'YES_CACHE_AND_VALIDATION','independent_new_dataset':'YES_NORMAN_RELATIVE_TO_TXPERT_AND_MCFALINE','main_confirmatory_qualified':False,'recommended_role':'SECONDARY_HISTORICAL_NATIVE_UQ','qualification_gap':'48 clusters; old truth already seen; condition holdout includes double-perturbation gene history; validation family RMSE gains modest and individual seed competence uneven'},
 {'asset':'Older_GEARS_7_frozen_panels','native_genes':'Adamson5043;Norman5025;Frangieh3000','persisted_checkpoints':0,'registered_evaluation_tasks':len(pairs),'distinct_evaluation_gene_clusters':len(set(genes)),'distinct_dataset_condition_clusters':len(set(pairs)),'registered_validation_conditions_per_panel':'5/20/18','expression_download_bytes':0,'new_training_gpu_hours':0,'can_run_within_one_day':'YES_EXISTING_PREDICTION_CACHE_ONLY','independent_new_dataset':'Adamson/Norman/Frangieh; Norman overlaps E195','main_confirmatory_qualified':False,'recommended_role':'HISTORICAL_MULTIDATASET_SENSITIVITY','qualification_gap':'3 datasets individually 71/48/48 genes; existing opened outcomes; absent saved checkpoints and task-level validation predictions; no new queries or fresh validation claim'},
 {'asset':'E201_TxPert_STRING_GAT_4targets_4seeds','native_genes':3352,'persisted_checkpoints':16,'registered_evaluation_tasks':2008,'primary_evaluation_tasks':1808,'target_primary_task_counts':'K562566;RPE1416;HepG2405;Jurkat421','registered_validation_conditions_per_model':457,'expression_download_bytes':0,'new_training_gpu_hours':0,'can_run_within_one_day':'YES_REUSE_SEALED_PREDICTIONS_AND_VALIDATION_LOGS','independent_new_dataset':False,'main_confirmatory_qualified':'EXISTING_ASSET_ONLY','recommended_role':'STRONG_EXISTING_GENETIC_CONTROL','qualification_gap':'Same TxPert/Replogle/Nadig assets already used by current main source; does not add independent external source; existing outcomes opened'},
 {'asset':'E190_GEARS_scGPT_Adamson_Replogle','native_genes':512,'persisted_checkpoints':'NOT_RECOUNTED_HERE','registered_evaluation_tasks':692,'distinct_evaluation_gene_clusters':47,'registered_validation_conditions_per_model':54,'expression_download_bytes':0,'new_training_gpu_hours':0,'can_run_within_one_day':'YES_EXISTING_RELEASE','independent_new_dataset':'PARTIAL_ADAMSON_HISTORY','main_confirmatory_qualified':False,'recommended_role':'CROSS_FAMILY_NARROW_PANEL_SENSITIVITY','qualification_gap':'512 native genes and 47 genetic clusters, 692 guide/batch records are not 692 independent perturbations'},
 {'asset':'ARC_VCC2025_H1_scPertEval_training_copy','native_genes':'NOT_DOWNLOADED_OR_OPENED','persisted_checkpoints':0,'registered_evaluation_tasks':0,'public_training_perturbations':150,'public_training_cells':221273,'public_control_cells':38176,'remote_h5ad_bytes':4377281643,'expression_download_bytes':0,'new_training_gpu_hours':0,'can_run_within_one_day':'NO_WITH_CURRENT_10MB_DOWNLOAD_LIMIT_AND_NO_VERIFIED_WEIGHTS','independent_new_dataset':True,'main_confirmatory_qualified':False,'recommended_role':'FUTURE_REGISTERED_DATASET','qualification_gap':'scPertEval copy is 150-gene training split and excludes official test; no verified inference checkpoint/prediction matrix; official main bucket requires subscribed project'},
 {'asset':'PRiMeFlow_Altos_official_VCC_winner_code','native_genes':18001,'persisted_checkpoints':0,'registered_evaluation_tasks':0,'official_private_test_prediction_spec_conditions':100,'remote_vcc_train_h5ad_gz_bytes':4291253394,'expression_download_bytes':0,'new_training_gpu_hours':0,'can_run_within_one_day':'NO_VERIFIED_WEIGHTS_OR_PREDICTION_MATRIX','independent_new_dataset':'H1_NEW_BUT_UNAVAILABLE_MODEL_ASSET','main_confirmatory_qualified':False,'recommended_role':'VERIFIED_PUBLIC_CODE_REFERENCE','qualification_gap':'Official code and data published; prediction_dataframe CSV is generation specification only; no model weights in HF dataset tree, no HF Altos models, no GitHub releases; official pretraining12GPUs and finetuning4GPUs200epochs'},
 {'asset':'scPertEval_models_public_scaffold','native_genes':'8192_HVG_union_perturbed_genes','persisted_checkpoints':0,'registered_evaluation_tasks':0,'published_models':6,'expression_download_bytes':0,'new_training_gpu_hours':0,'can_run_within_one_day':'NO_VERIFIED_COMPLETED_MODEL_OUTPUT_RELEASE','independent_new_dataset':False,'main_confirmatory_qualified':False,'recommended_role':'TRAINING_SCAFFOLD_REFERENCE','qualification_gap':'Verified scripts require train_predict per fold; public bucket list contains processed expression and DE outputs only, no prediction matrices; Replogle source overlaps current assets'}
]
pd.DataFrame(candidates).to_csv(OUT/'ASSET_AGENT_AUDIT.candidates.csv',index=False)
audit={'date':'2026-10-01','selection_rule':'Qualification by persisted architecture, output dimension, legal validation access, independence and runtime; no SafeConf score used to select candidates','selected_asset_by_agent':None,'decision_owner':'root','candidates':candidates,'old_gears_clusters':{'task_records':len(pairs),'dataset_condition_clusters':len(set(pairs)),'gene_clusters':len(set(genes)),'per_dataset':{d:len({c for dd,c in pairs if dd==d}) for d in ['adamson','norman','frangieh']}},'validation_competence_scope':'registered E195 validation; all six frozen checkpoints; no hyperparameter fitting or seed selection','validation_access':json.loads((OUT/'ASSET_AGENT_AUDIT.validation_access.json').read_text()),'E195_validation_family':pd.read_csv(OUT/'ASSET_AGENT_AUDIT.validation_family_summary.csv').to_dict(orient='records'),'E201_existing_checkpoint_hashes_verified':len(tx),'network_access':'HTML/README/listing/HEAD metadata only; no expression or checkpoint download; no account or billing action','no_new_test_truth_or_expression_accessed':True}
(OUT/'ASSET_AGENT_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps({'candidates':len(candidates),'E201_checkpoint_hashes_verified':len(tx),'old_GEARSprediction_files':sum(bool(x['prediction_exists']) for x in panels),'old_gene_clusters':len(set(genes)),'E195_validation_family':audit['E195_validation_family']},ensure_ascii=False))

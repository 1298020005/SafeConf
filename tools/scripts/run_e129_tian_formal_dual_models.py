#!/usr/bin/env python3
"""E129: formal scGPT/GEARS predictions on the frozen E128 Tian contract."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from datetime import datetime
from pathlib import Path
import pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2];SOURCE_SCRIPT=ROOT/'tools/scripts/run_e112_external_formal_dual_models.py';CONTRACT=ROOT/'docs/实验结果/E128_tian_crispri_four_batch_contract_20260714/manifests/E128_TASK_MANIFEST.csv';OUT=ROOT/'docs/实验结果/E129_tian_crispri_formal_dual_models_20260714'
def module():
 s=importlib.util.spec_from_file_location('e112_for_e129',SOURCE_SCRIPT);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);m.CONTRACT=CONTRACT;m.OUT=OUT;m.SPECS={'Tian_CRISPRi':{'source':Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/TianKampmann2021_CRISPRi.h5ad'),'context':'batch'}};return m
def finalize():
 root=OUT/'Tian_CRISPRi';tasks=pd.read_csv(root/'TASK_RISK_TABLE.csv');s=json.loads((root/'RUN_STATUS.json').read_text());status={'experiment':'E129_tian_crispri_formal_dual_models','generated_at':datetime.now().isoformat(timespec='seconds'),'status':s['status'],'source_contract':str(CONTRACT.relative_to(ROOT)),'dataset':'TianKampmann2021_CRISPRi','context_definition':'technical batch shift','n_folds':int(tasks.fold_id.nunique()),'n_test_tasks':len(tasks),'n_prediction_records':int(s['n_records']),'strict_issue_count':int(s['strict_issue_count']),'test_truth_used_for_training_calibration_score_or_threshold':False};(OUT/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
 rows=[]
 for setting,g in tasks.groupby('setting'):
  for score in ['safeconf_calibrated_pair_risk','safeconf_frozen_pair_risk','risk_model_disagreement','baseline_predicted_magnitude']:rows.append({'setting':setting,'score':score,'spearman':float(g[[score,'error_two_predictor_mean_rmse']].corr(method='spearman').iloc[0,1])})
 pd.DataFrame(rows).to_csv(OUT/'E129_SETTING_SUMMARY.csv',index=False);(OUT/'E129_REPORT.md').write_text(f"# E129｜Tian–Kampmann CRISPRi 正式双模型验证\n\n严格记录：{status['n_prediction_records']}；issues={status['strict_issue_count']}。四个背景是技术批次，结果只用于技术域偏移与 E127 路由器的第六数据确认。\n");(OUT/'README_先看这个.md').write_text('# E129 先看这个\n\n先读 `E129_REPORT.md`。\n');print(json.dumps(status,ensure_ascii=False,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--finalize-only',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if not a.finalize_only:m=module();device=torch.device(a.device if torch.cuda.is_available() else 'cpu');print(json.dumps(m.run_dataset('Tian_CRISPRi',device),ensure_ascii=False,indent=2))
 finalize()
if __name__=='__main__':main()

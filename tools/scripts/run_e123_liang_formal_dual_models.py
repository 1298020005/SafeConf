#!/usr/bin/env python3
"""E123: formal scGPT/GEARS prediction on untouched Liang E122 tasks."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from datetime import datetime
from pathlib import Path
import pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2];SOURCE_SCRIPT=ROOT/'tools/scripts/run_e112_external_formal_dual_models.py';CONTRACT=ROOT/'docs/实验结果/E122_liang_nine_context_contract_20260714/manifests/E122_TASK_MANIFEST.csv';OUT=ROOT/'docs/实验结果/E123_liang_formal_dual_models_20260714'
def load_e112():
 spec=importlib.util.spec_from_file_location('e112_for_e123',SOURCE_SCRIPT);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);m.CONTRACT=CONTRACT;m.OUT=OUT;m.SPECS={'Liang':{'source':Path('/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/LiangWang2023.h5ad'),'context':'sample'}};m.ALIASES.update({'Sfpi1':'SPI1'});return m
def finalize():
 root=OUT/'Liang';tasks=pd.read_csv(root/'TASK_RISK_TABLE.csv');s=json.loads((root/'RUN_STATUS.json').read_text());status={'experiment':'E123_liang_formal_dual_models','generated_at':datetime.now().isoformat(timespec='seconds'),'status':s['status'],'source_contract':str(CONTRACT.relative_to(ROOT)),'dataset':'LiangWang2023','context_definition':'organoid culture replicate nested in differentiation/expansion medium','n_folds':int(tasks.fold_id.nunique()),'n_test_tasks':len(tasks),'n_prediction_records':int(s['n_records']),'strict_issue_count':int(s['strict_issue_count']),'test_truth_used_for_training_calibration_score_or_threshold':False};(OUT/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');summary=pd.read_csv(root/'SETTING_SUMMARY.csv').groupby(['setting','score'],as_index=False).spearman.mean();pooled=[]
 for score in ['safeconf_calibrated_pair_risk','safeconf_frozen_pair_risk','risk_model_disagreement','baseline_predicted_magnitude']:
  values=[g[[score,'error_two_predictor_mean_rmse']].corr(method='spearman').iloc[0,1] for _,g in tasks.groupby('fold_id')];pooled.append({'setting':'all_test_settings_pooled','score':score,'spearman':float(pd.Series(values).mean())})
 summary=pd.concat([summary,pd.DataFrame(pooled)],ignore_index=True);summary.to_csv(OUT/'E123_SETTING_MACRO.csv',index=False);lines=['# E123｜Liang–Wang 九背景正式双模型复制','','该数据合同在 Shifrut 结果解封以后冻结，但测试真值在本轮预测、风险落盘前未读取。scGPT 和 GEARS 沿用 E112/E120 的固定训练与输入合同。','',f"- folds：{status['n_folds']}",f"- test tasks：{status['n_test_tasks']}",f"- strict records：{status['n_prediction_records']}；issues={status['strict_issue_count']}",'','| setting | score | macro Spearman |','|---|---|---:|']
 for r in summary.itertuples(index=False):lines.append(f'| {r.setting} | {r.score} | {r.spearman:.3f} |')
 (OUT/'E123_REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'README_先看这个.md').write_text('# E123 先看这个\n\n先读 `E123_REPORT.md`。\n');print(json.dumps(status,ensure_ascii=False,indent=2));print(summary.to_string(index=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--finalize-only',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if not a.finalize_only:m=load_e112();print(json.dumps(m.run_dataset('Liang',torch.device(a.device if torch.cuda.is_available() else 'cpu')),ensure_ascii=False,indent=2))
 finalize()
if __name__=='__main__':main()

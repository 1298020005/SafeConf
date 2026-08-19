#!/usr/bin/env python3
"""E132: descriptive six-dataset triage utility after Tian unblinding."""
from __future__ import annotations
import importlib.util,json,sys
from datetime import datetime
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'tools/scripts/run_e115_formal_triage_utility.py';OUT=ROOT/'docs/实验结果/E132_six_dataset_triage_utility_20260714';TABLES=OUT/'tables';REPORTS=OUT/'reports';FIGURES=OUT/'figures'
def mod():
 s=importlib.util.spec_from_file_location('e115_for_e132',BASE);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);m.OUT=OUT;m.TABLES=TABLES;m.REPORTS=REPORTS;m.FIGURES=FIGURES;return m
def load():
 a=pd.read_csv(ROOT/'docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv');a['dataset']='Frangieh';b=pd.read_csv(ROOT/'docs/实验结果/E112_external_formal_dual_models_20260713/E112_ALL_TASKS.csv');c=pd.read_csv(ROOT/'docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut/TASK_RISK_TABLE.csv');c['dataset']='Shifrut';d=pd.read_csv(ROOT/'docs/实验结果/E123_liang_formal_dual_models_20260714/Liang/TASK_RISK_TABLE.csv');d['dataset']='Liang';e=pd.read_csv(ROOT/'docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi/TASK_RISK_TABLE.csv');e['dataset']='Tian_CRISPRi';return pd.concat([a,b,c,d,e],ignore_index=True,sort=False)
def main():
 for d in (TABLES,REPORTS,FIGURES):d.mkdir(parents=True,exist_ok=True)
 m=mod();data=load();folds,curves=m.compute(data);summary=m.macro(folds);summary['scope']=summary.scope.replace({'three_dataset_equal_macro':'six_dataset_equal_macro'});boot=m.bootstrap(folds);folds.to_csv(TABLES/'E132_FOLD_METRICS.csv',index=False);curves.to_csv(TABLES/'E132_RISK_COVERAGE_CURVES.csv',index=False);summary.to_csv(TABLES/'E132_MACRO_SUMMARY.csv',index=False);boot.to_csv(TABLES/'E132_BOOTSTRAP.csv',index=False);point=summary[summary.scope.eq('six_dataset_equal_macro')].set_index('score');lines=['# E132｜六正式数据集的分诊效用','','E132 在 Tian 解封后追加，属于完整性描述；E127/E130 才是 Tian 事前冻结的路由器检验。','','| score | normalized AURC↓ | top-20% enrichment↑ | reject-20% reduction↑ | top-20% capture↑ |','|---|---:|---:|---:|---:|']
 for score,r in point.iterrows():lines.append(f'| {score} | {r.normalized_aurc_50_100:.4f} | {r.top20_error_enrichment:.4f} | {r.reject20_remaining_error_reduction:.4f} | {r.top20_total_error_capture:.4f} |')
 lines+=['','| metric | comparator | favorable Δ | 95% CI | P(Δ>0) |','|---|---|---:|---:|---:|']
 for r in boot.itertuples(index=False):lines.append(f'| {r.metric} | {r.comparator} | {r.favorable_delta:.4f} | [{r.ci95_low:.4f}, {r.ci95_high:.4f}] | {r.probability_favorable:.3f} |')
 (REPORTS/'E132_REPORT.md').write_text('\n'.join(lines)+'\n');status={'experiment':'E132_six_dataset_triage_utility','generated_at':datetime.now().isoformat(timespec='seconds'),'status':'complete','n_datasets':int(data.dataset.nunique()),'n_folds':int(data.fold_id.nunique()),'n_tasks':len(data),'confirmatory':False,'risk_scores_refit_on_test_truth':False};(OUT/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');(OUT/'README_先看这个.md').write_text('# E132 先看这个\n\n先读 `reports/E132_REPORT.md`。\n');print(json.dumps(status,ensure_ascii=False,indent=2));print(point.to_string());print(boot.to_string(index=False))
if __name__=='__main__':main()

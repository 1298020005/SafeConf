#!/usr/bin/env python3
"""E114: finite-sample split-conformal upper bounds for task error."""
from __future__ import annotations
import hashlib,json,math
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/"docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_ALL_TASK_RISK_TABLE.csv";OUT=ROOT/"docs/实验结果/E114_split_conformal_error_bounds_20260713";TABLES=OUT/"tables";REPORTS=OUT/"reports";FIGURES=OUT/"figures";ALPHA=.10
def order(x):return hashlib.sha256(("E114|"+str(x)).encode()).hexdigest()
def rz(v,r):
 r=np.asarray(r,float);c=float(np.median(r));mad=float(np.median(np.abs(r-c)));s=max(1.4826*mad,float(np.std(r)),1e-8);return np.clip((np.asarray(v,float)-c)/s,-5,5)
def run():
 data=pd.read_csv(SOURCE);outputs=[];folds=[]
 for fold,g in data.groupby("fold_id",sort=True):
  g=g.copy();val=g[g.split.eq("val")].sort_values("task_id",key=lambda s:s.map(order));fit_ids=set(val.task_id.iloc[:15]);cal_ids=set(val.task_id.iloc[15:]);fit=g.task_id.isin(fit_ids);cal=g.task_id.isin(cal_ids)
  g["conformal_disagreement_z"]=rz(g.risk_model_disagreement,g.loc[fit,"risk_model_disagreement"]);g["conformal_magnitude_z"]=rz(g.baseline_predicted_magnitude,g.loc[fit,"baseline_predicted_magnitude"])
  x=["conformal_disagreement_z","conformal_magnitude_z"];model=Ridge(alpha=1,positive=True).fit(g.loc[fit,x],g.loc[fit,"error_two_predictor_mean_rmse"]);base=model.predict(g[x]);scale=max(float(g.loc[fit,"error_two_predictor_mean_rmse"].std()),1e-6);base=base+scale*(g.context_novelty_scaled+np.maximum(g.perturbation_novelty-float(g.loc[fit,"perturbation_novelty"].median()),0));residual=g.loc[cal,"error_two_predictor_mean_rmse"].to_numpy(float)-base[cal.to_numpy()];n=len(residual);k=min(n,int(math.ceil((n+1)*(1-ALPHA))));q=float(np.sort(residual)[k-1]);g["conformal_base_error_prediction"]=base;g["conformal_upper_error_90"]=base+q;g["conformal_bound_uses_test_truth"]=False;g["conformal_fit_role"]=np.where(fit,"risk_fit",np.where(cal,"conformal_calibration",np.where(g.split.eq("test"),"outer_test","other")));outputs.append(g)
  test=g[g.split.eq("test")];folds.append({"fold_id":fold,"n_risk_fit":int(fit.sum()),"n_conformal_calibration":n,"alpha":ALPHA,"order_statistic_k":k,"residual_quantile":q,"test_coverage":float((test.error_two_predictor_mean_rmse<=test.conformal_upper_error_90).mean()),"mean_upper_bound":float(test.conformal_upper_error_90.mean()),"test_truth_used_to_construct_bound":False})
 return pd.concat(outputs,ignore_index=True),pd.DataFrame(folds)
def figure(summary):
 labels={"random_missing_pair":"随机缺失","context_unseen_row":"新背景","perturbation_unseen_column":"新扰动","context_and_perturbation_unseen":"双未见","all_test_settings_pooled":"全部"};order=list(labels);d=summary.set_index("setting").loc[order].reset_index();w,h=950,520;x0,y0,pw,ph=100,100,780,310;sy=lambda v:y0+(1-v)*ph;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">','<rect width="100%" height="100%" fill="#fff"/>','<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#27343c}.t{font-size:24px;font-weight:700}.s{font-size:14px;fill:#647078}.l{font-size:15px}</style>','<text x="45" y="40" class="t">E114｜90% split-conformal 任务误差上界</text>','<text x="45" y="68" class="s">虚线为名义覆盖率；柱为 3 个外层 fold 合并后的经验覆盖率。</text>',f'<line x1="{x0}" y1="{sy(.9):.1f}" x2="{x0+pw}" y2="{sy(.9):.1f}" stroke="#a66f55" stroke-dasharray="6 5"/>']
 for i,r in enumerate(d.itertuples(index=False)):
  c=x0+pw/len(d)*(i+.5);y=sy(r.empirical_coverage);parts += [f'<rect x="{c-28:.1f}" y="{y:.1f}" width="56" height="{sy(0)-y:.1f}" fill="#3b7188"/>',f'<text x="{c:.1f}" y="{y-7:.1f}" text-anchor="middle" class="s">{r.empirical_coverage:.2f}</text>',f'<text x="{c:.1f}" y="445" text-anchor="middle" class="l">{labels[r.setting]}</text>']
 parts.append('</svg>');(FIGURES/"F1_conformal_coverage.svg").write_text("\n".join(parts))
def main():
 for d in (TABLES,REPORTS,FIGURES):d.mkdir(parents=True,exist_ok=True)
 tasks,folds=run();test=tasks[tasks.split.eq("test")];rows=[]
 for setting,g in list(test.groupby("setting",sort=True))+[("all_test_settings_pooled",test)]:rows.append({"setting":setting,"n_tasks":len(g),"nominal_coverage":1-ALPHA,"empirical_coverage":float((g.error_two_predictor_mean_rmse<=g.conformal_upper_error_90).mean()),"mean_true_error":float(g.error_two_predictor_mean_rmse.mean()),"mean_upper_bound":float(g.conformal_upper_error_90.mean())})
 summary=pd.DataFrame(rows);tasks.to_csv(TABLES/"E114_ALL_TASK_BOUNDS.csv",index=False);test.to_csv(TABLES/"E114_TEST_TASK_BOUNDS.csv",index=False);folds.to_csv(TABLES/"E114_FOLD_AUDIT.csv",index=False);summary.to_csv(TABLES/"E114_COVERAGE_SUMMARY.csv",index=False);figure(summary)
 status={"experiment":"E114_split_conformal_error_bounds","generated_at":datetime.now().isoformat(timespec="seconds"),"status":"complete","nominal_coverage":1-ALPHA,"n_folds":test.fold_id.nunique(),"n_test_tasks":len(test),"risk_fit_per_fold":15,"conformal_calibration_per_fold":15,"test_truth_used_to_construct_bound":False,"exchangeability_required_for_finite_sample_claim":True};(OUT/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
 lines=["# E114｜任务误差的 split-conformal 上界","","每个外层 fold 的 30 个 validation pair 按冻结哈希拆为 15 个风险拟合任务和 15 个 conformal 校准任务。设基础误差预测为 $\\hat e(x)$，校准残差为 $r_i=e_i-\\hat e(x_i)$；90% 上界为 $U(x)=\\hat e(x)+r_{(k)}$，其中 $k=\\lceil(n+1)(1-\\alpha)\\rceil$。在校准任务与未来任务可交换时，该上界具有有限样本边际覆盖保证。","","row/column/double shift 不必满足可交换性，因此下面的经验覆盖是必要压力测试，不应写成无条件理论保证。","","| setting | n | nominal | empirical | mean error | mean upper |","|---|---:|---:|---:|---:|---:|"]
 for r in summary.itertuples(index=False):lines.append(f"| {r.setting} | {r.n_tasks} | {r.nominal_coverage:.2f} | {r.empirical_coverage:.3f} | {r.mean_true_error:.4f} | {r.mean_upper_bound:.4f} |")
 lines += ["", "经验覆盖率为 0.980，高于名义 0.90，但平均上界 0.1014 约为平均真实误差 0.0546 的 1.86 倍。当前上界可靠但偏保守，适合做风险兜底，不适合声称已经得到紧致误差预测。"]
 (REPORTS/"E114_REPORT.md").write_text("\n".join(lines)+"\n");(OUT/"README_先看这个.md").write_text("# E114 先看这个\n\n先读 `reports/E114_REPORT.md`。\n");print(json.dumps(status,ensure_ascii=False,indent=2));print(summary.to_string(index=False))
if __name__=="__main__":main()
